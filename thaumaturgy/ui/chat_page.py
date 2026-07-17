"""Chat page — conversation view with a per-scenario chat sidebar.

Left sidebar: scenario selector, "New chat", and the list of previous chats for
the selected scenario (chats are tied to scenarios, persisted as JSON). Main
area: the transcript + input. Send streams the assistant's reply live from the
llama-server engine using the selected scenario + parameter set.
"""

import asyncio
import re
import threading
import time

from nicegui import app, run, ui

from thaumaturgy import appstate, engine, store

# Matches both dialects: `<|channel>thought` and `<|channel|>analysis<|message|>`.
# The terminator is required so a name still streaming in ("<|channel>a") isn't
# read as complete.
_CHANNEL_MARKER = "<|channel"
_CHANNEL_RE = re.compile(
    r"<\|channel\|?>[ \t]*([A-Za-z0-9_.-]+)[ \t]*(?:<\|message\|>|<channel\|>|\r?\n)")
_CONTROL_RE = re.compile(
    r"<\|start\|>[ \t]*assistant|<\|(?:start|end|return|message)\|>|<channel\|>")
_THOUGHT_CHANNELS = {"thought", "thinking", "reasoning", "analysis"}


def _rel_time(ts: float | None) -> str:
    if not ts:
        return ""
    delta = max(0, time.time() - ts)
    if delta < 60:
        return "just now"
    for unit, secs in (("d", 86400), ("h", 3600), ("m", 60)):
        if delta >= secs:
            return f"{int(delta // secs)}{unit} ago"
    return "just now"


def _avatar(m: dict):
    is_user = m["role"] == "user"
    with ui.avatar(color="primary" if is_user else "secondary").props("text-color=white"):
        ui.label((m.get("name") or "?")[0].upper())


def _finish_warning(reason: str | None) -> str | None:
    if not reason or reason == "stop":
        return None
    if reason == "error":
        return "Generation failed before the model finished replying."
    if reason == "length":
        return "Generation stopped because Max new tokens was reached."
    return f"Generation finished with reason: {reason}."


def _message_warning(m: dict) -> str | None:
    if m.get("generation_error"):
        return f"Generation failed: {m['generation_error']}"
    return _finish_warning(m.get("finish_reason"))


def _join_blocks(parts: list[str]) -> str:
    return "\n\n".join(part.strip() for part in parts if part and part.strip()).strip()


def _split_reasoning_channels(text: str) -> tuple[str, str]:
    """Split channel-marked output into visible text and reasoning text.

    Needed for templates llama.cpp can't parse (Gemma's): their markers arrive
    raw in the reply rather than as reasoning events.

    Both halves are stripped: ui.markdown measures indentation from the first
    non-empty line and slices that many chars off every line, so a reply opening
    with llama.cpp's usual leading space loses a character per line below it.
    """
    if not text or _CHANNEL_MARKER not in text:
        return text.strip(), ""
    matches = list(_CHANNEL_RE.finditer(text))
    if not matches:
        return text.strip(), ""

    visible_parts = [_CONTROL_RE.sub("", text[:matches[0].start()])]
    reasoning_parts = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = _CONTROL_RE.sub("", text[match.end():end])
        target = (reasoning_parts if match.group(1).lower() in _THOUGHT_CHANNELS
                  else visible_parts)
        target.append(content)
    return _join_blocks(visible_parts), _join_blocks(reasoning_parts)


def _visible_and_reasoning(text: str, reasoning: str) -> tuple[str, str]:
    """Promote reasoning to the reply when the model produced nothing else.

    Some models put ordinary prose in the thought channel and never open a final
    one; the bubble would otherwise be empty.
    """
    if text.strip():
        return text, reasoning
    return reasoning, ""


def _message_text_and_reasoning(m: dict) -> tuple[str, str]:
    text = m.get("text") or ""
    reasoning = (m.get("reasoning") or "").strip()
    if m.get("role") == "assistant":
        text, marker_reasoning = _split_reasoning_channels(text)
        reasoning = reasoning or marker_reasoning
    return _visible_and_reasoning(text, reasoning)


class _MessageView:
    """Live handles into one rendered message, so a stream can update it in place.

    The Thinking pane is built up front and hidden because the bubble's slot is
    closed by the time reasoning arrives — an observer can't add elements then.
    """

    def __init__(self, text_md, reasoning_box=None, reasoning_md=None):
        self.text_md = text_md
        self.reasoning_box = reasoning_box
        self.reasoning_md = reasoning_md

    @property
    def is_deleted(self) -> bool:
        return self.text_md.is_deleted

    def update(self, text: str, reasoning: str) -> None:
        self.text_md.content = text
        if self.reasoning_box is None:
            return
        self.reasoning_md.content = reasoning
        self.reasoning_box.set_visibility(bool(reasoning.strip()))


def _message(m: dict, on_scenario_click=None) -> _MessageView:
    """Render one message row; returns handles to it (for live updates)."""
    is_user = m["role"] == "user"
    clickable = (not is_user) and on_scenario_click is not None
    with ui.row().classes("w-full gap-3 no-wrap items-start pb-4"):
        col = ui.column().classes("items-center gap-1 w-16 shrink-0")
        if clickable:
            col.classes("cursor-pointer hover:opacity-80")
            col.on("click", lambda: on_scenario_click(m.get("model")))
        with col:
            _avatar(m)
            ui.label(m.get("name") or "").classes(
                "text-xs text-center leading-tight "
                + ("text-primary" if clickable else "text-muted"))
        bubble = ui.column().classes("flex-1 min-w-0 gap-1 p-3 rounded-xl")
        if is_user:
            bubble.style("background: rgba(52,97,140,0.10)")
        with bubble:
            text, reasoning = _message_text_and_reasoning(m)
            md = ui.markdown(text).classes("text-sm leading-relaxed break-words")
            box = reasoning_md = None
            if not is_user:
                box = ui.expansion("Thinking", icon="psychology").classes("w-full")
                with box:
                    reasoning_md = ui.markdown(reasoning).classes(
                        "text-xs leading-relaxed break-words text-muted")
                box.set_visibility(bool(reasoning))
            warning = _message_warning(m)
            if warning:
                ui.badge(warning).props("color=warning text-color=dark") \
                    .classes("self-start text-xs mt-1")
    return _MessageView(md, box, reasoning_md)


def _start_generation(chat: dict, api: list[dict], assistant: dict, params: dict) -> dict:
    """Run model streaming off the UI task and save partial output as it arrives."""
    chat_id = chat["id"]
    state = {
        "chat_id": chat_id,
        "chat": chat,
        "assistant": assistant,
        "assistant_index": len(chat.get("messages", [])) - 1,
        "done": False,
        "error": None,
    }
    appstate.state.generations[chat_id] = state

    def worker():
        last_save = 0.0
        raw_text = assistant.get("text", "")
        raw_reasoning = assistant.get("reasoning", "")
        try:
            for event in engine.server.stream_chat(api, params):
                kind = event.get("type")
                if kind == "finish":
                    assistant["finish_reason"] = event.get("reason")
                    continue
                delta = event.get("text", "")
                if not delta:
                    continue
                if kind == "reasoning":
                    raw_reasoning += delta
                else:
                    raw_text += delta
                text, marker_reasoning = _split_reasoning_channels(raw_text)
                text, reasoning = _visible_and_reasoning(
                    text, _join_blocks([raw_reasoning, marker_reasoning]))
                assistant["text"] = text
                if reasoning:
                    assistant["reasoning"] = reasoning
                else:
                    assistant.pop("reasoning", None)
                now = time.monotonic()
                if now - last_save > 0.5:
                    store.save_chat(chat)
                    last_save = now
        except Exception as exc:  # noqa: BLE001 - stored for the observing UI
            error = str(exc)
            state["error"] = error
            assistant["finish_reason"] = "error"
            assistant["generation_error"] = error
        finally:
            store.save_chat(chat)
            if appstate.state.generations.get(chat_id) is state:
                del appstate.state.generations[chat_id]
            state["done"] = True  # last: observers read the chat back off disk

    threading.Thread(target=worker, daemon=True).start()
    return state


def _prepend_to_first_user(messages: list[dict], prefix: str) -> list[dict]:
    for msg in messages:
        if msg.get("role") == "user":
            msg["content"] = f"{prefix}\n\n{msg.get('content', '')}".strip()
            return messages
    if prefix:
        messages.insert(0, {"role": "user", "content": prefix})
    return messages


def _api_messages(chat: dict, scenario: str, scenario_details: dict,
                  draft: str = "") -> list[dict]:
    """Build the /v1/chat/completions message list, including any unsent `draft`.

    The draft is folded in here, not appended by the caller: without a system
    role the scenario merges into the first user turn, which may be the draft.
    """
    details = scenario_details.get(scenario, {})
    system_parts = []
    context = (details.get("context") or "").strip()
    if context:
        system_parts.append(context)
    chat_messages = list(chat.get("messages", []))
    # Gemma-style templates raise on a leading assistant turn, and no
    # chat_template_caps flag reports it — so the opening always moves to the prompt.
    while chat_messages and chat_messages[0].get("role") != "user":
        opening = (chat_messages.pop(0).get("text") or "").strip()
        if opening:
            system_parts.append(f"Opening scene:\n{opening}")
    messages = []
    for m in chat_messages:
        text = (m.get("text") or "").strip()
        if not text and m.get("generation_error"):
            continue
        role = "user" if m["role"] == "user" else "assistant"
        messages.append({"role": role, "content": m.get("text", "")})
    if draft:
        messages.append({"role": "user", "content": draft})
    system_content = "\n\n".join(system_parts).strip()
    if not system_content:
        return messages
    if engine.server.supports_system_role():
        return [{"role": "system", "content": system_content}, *messages]
    return _prepend_to_first_user(messages, system_content)


def _context_total(model_name: str | None = None) -> int | None:
    if engine.server.n_ctx:
        return engine.server.n_ctx
    model = model_name or engine.server.model or appstate.state.current_model
    return engine.trained_ctx(model) if model else None


def _context_label(used: int | None, total: int | None, exact: bool = True) -> str:
    if used is None:
        return "Context --"
    prefix = "" if exact else "~"
    if total:
        pct = min(999, round((used / total) * 100))
        return f"Context {prefix}{used:,} / {total:,} ({pct}%)"
    return f"Context {prefix}{used:,}"


def _truncate(text: str, max_len: int = 40) -> str:
    return text if len(text) <= max_len else f"{text[:max_len]}..."


def _normalize_user_markdown(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    return re.sub(r"(?<!\n)\n(?!\n)", "\n\n", text)


def _latest_assistant_response_index(messages: list[dict]) -> int | None:
    if not messages or messages[-1].get("role") != "assistant":
        return None
    if not any(m.get("role") == "user" for m in messages[:-1]):
        return None
    return len(messages) - 1


def render():
    """Build the Chat page inside the current layout container."""
    scenarios = store.list_scenarios()
    scenario_names = [s["name"] for s in scenarios]
    scenario_details = {s["name"]: s for s in scenarios}
    if appstate.state.current_scenario not in scenario_names:
        appstate.state.current_scenario = scenario_names[0] if scenario_names else None
    page = {"chat": None, "refresh_context": lambda: None}

    def generation_for_chat(chat_id: str | None) -> dict | None:
        return appstate.state.generations.get(chat_id or "")

    def load_chat_state(chat_id: str | None) -> dict | None:
        """Load a chat, preferring the live copy an in-flight generation writes to."""
        generation = generation_for_chat(chat_id)
        if generation:
            return generation["chat"]
        return store.load_chat(chat_id) if chat_id else None

    def first_chat(scenario: str | None) -> dict | None:
        chats = store.list_chats(scenario)
        return load_chat_state(chats[0]["id"]) if chats else None

    # ── Scenario info panel (slides in from the right) ───────────────────────
    backdrop = ui.element("div").classes("tg-backdrop")
    info_panel = ui.column().classes("tg-slidepanel p-4 gap-3")

    def close_panel():
        info_panel.classes(remove="tg-open")
        backdrop.classes(remove="tg-open")

    backdrop.on("click", close_panel)
    with info_panel:
        with ui.row().classes("w-full justify-start"):
            ui.button(icon="close", on_click=close_panel).props("flat round dense")
        detail = ui.column().classes("w-full gap-3 items-center")

    def open_scenario(model: str | None = None):
        name = appstate.state.current_scenario
        scenario = scenario_details.get(
            name, {"name": name or "", "context": "", "opening_text": ""})
        detail.clear()
        with detail:
            with ui.column().classes("w-full gap-0 items-center"):
                ui.label("MODEL").classes("text-xs text-muted tracking-wide")
                model_name = model or "unknown"
                ui.badge(_truncate(model_name)).props("color=primary").classes(
                    "text-[10px] font-mono text-center break-all max-w-full").tooltip(model_name)
            ui.separator()
            with ui.avatar(color="secondary", size="88px").props("text-color=white"):
                ui.label((scenario["name"] or "?")[0].upper()).classes("text-3xl")
            ui.label(scenario["name"]).classes("text-lg font-semibold text-center")
            ui.separator()
            with ui.column().classes("w-full gap-1"):
                ui.label("SCENARIO CONTEXT").classes("text-xs text-muted tracking-wide")
                ui.markdown(scenario["context"] or "_None_").classes("text-sm leading-relaxed")
                ui.label("OPENING TEXT").classes("text-xs text-muted tracking-wide mt-3")
                ui.markdown(scenario["opening_text"] or "_None_").classes("text-sm leading-relaxed")
        info_panel.classes(add="tg-open")
        backdrop.classes(add="tg-open")

    # ── Transcript rendering (container-based so we can stream into one msg) ──
    def render_reply_actions():
        with ui.row().classes("w-full gap-2 no-wrap items-start pb-2"):
            ui.element("div").classes("w-16 shrink-0")
            ui.button("Regenerate", icon="refresh", on_click=regenerate_last) \
                .props("flat dense color=secondary") \
                .classes("text-xs")
            ui.button("Edit", icon="edit", on_click=edit_last_response) \
                .props("flat dense color=secondary") \
                .classes("text-xs")

    def render_messages():
        msgs_col.clear()
        page["inner"] = None
        page["stream_view"] = None
        with msgs_col:
            chat = page["chat"]
            if not chat:
                with ui.column().classes("w-full h-full items-center justify-center gap-2"):
                    ui.icon("forum").classes("text-5xl text-muted")
                    ui.label("Start a new chat.").classes("text-muted")
                return
            inner = ui.column().classes("w-full max-w-3xl mx-auto gap-2")
            page["inner"] = inner
            with inner:
                generation = generation_for_chat(chat.get("id"))
                regenerate_index = (
                    None if generation
                    else _latest_assistant_response_index(chat.get("messages", []))
                )
                for i, m in enumerate(chat["messages"]):
                    view = _message(m, on_scenario_click=open_scenario)
                    if generation and i == generation["assistant_index"]:
                        page["stream_view"] = view
                    if i == regenerate_index:
                        render_reply_actions()

    def scroll_bottom():
        transcript_scroll.scroll_to(percent=1.0)

    async def scroll_bottom_after_render():
        await asyncio.sleep(0.05)
        if not transcript_scroll.is_deleted:
            scroll_bottom()

    def observing(generation: dict) -> bool:
        """True while this page is still showing the chat this generation feeds."""
        if msgs_col.is_deleted or transcript_scroll.is_deleted:
            return False
        return bool(page["chat"]) and page["chat"].get("id") == generation["chat_id"]

    async def observe_generation(generation: dict):
        """Mirror a running generation into the transcript until it finishes."""
        last = None
        try:
            while not generation["done"]:
                await asyncio.sleep(0.1)
                if not observing(generation):
                    return
                view = page.get("stream_view")
                if view is None or view.is_deleted:
                    return
                assistant = generation["assistant"]
                current = (assistant.get("text", ""), assistant.get("reasoning", ""))
                if current != last:
                    view.update(*current)
                    scroll_bottom()
                    last = current

            if not observing(generation):
                return
            assistant = generation["assistant"]
            view = page.get("stream_view")
            if view is not None and not view.is_deleted:
                view.update(assistant.get("text") or "_(no output)_",
                            assistant.get("reasoning", ""))
            if _message_warning(assistant):
                render_messages()  # re-render to hang the warning badge off the bubble
            elif (
                _latest_assistant_response_index(generation["chat"].get("messages", []))
                == generation["assistant_index"]
            ):
                with page["inner"]:
                    render_reply_actions()
            scroll_bottom()
            chat_list.refresh()
            if generation["error"]:
                # Runs as a bare task (see watch_generation), which has no slot
                # stack of its own — ui.notify needs one to find the client.
                with msgs_col:
                    ui.notify(f"Generation error: {generation['error']}", type="negative")
        finally:
            if page.get("observed") is generation:
                page["observed"] = None

    def watch_generation(chat_id: str | None):
        """Start observing the generation feeding `chat_id`, if there is one."""
        generation = generation_for_chat(chat_id)
        if generation and page.get("observed") is not generation:
            page["observed"] = generation  # at most one observer at a time
            asyncio.create_task(observe_generation(generation))

    # ── Chat management ──────────────────────────────────────────────────────
    def show_chat(chat: dict | None):
        """Make `chat` the one this page displays, and resume watching its stream."""
        page["chat"] = chat
        appstate.state.current_chat_id = chat["id"] if chat else None
        render_messages()
        chat_list.refresh()
        watch_generation(chat["id"] if chat else None)
        page["refresh_context"]()
        asyncio.create_task(scroll_bottom_after_render())

    def load_chat(chat_id: str):
        show_chat(load_chat_state(chat_id))

    pending_delete = {"chat": None}
    pending_edit = {"chat": None, "index": None}

    with ui.dialog() as delete_dialog, ui.card().classes("p-5 gap-3").style("width:420px;max-width:92vw"):
        delete_label = ui.label().classes("text-sm leading-relaxed")
        with ui.row().classes("w-full justify-end gap-2"):
            ui.button("Cancel", on_click=delete_dialog.close).props("flat")
            ui.button("Delete", icon="delete",
                      on_click=lambda: (delete_dialog.close(), delete_pending_chat())) \
                .props("color=negative unelevated")

    with ui.dialog() as edit_dialog, ui.card().classes("p-5 gap-3").style("width:720px;max-width:92vw"):
        ui.label("Edit Response").classes("text-lg font-semibold")
        edit_box = ui.textarea() \
            .props("filled autogrow input-style=max-height:60vh") \
            .classes("w-full tg-field")
        with ui.row().classes("w-full justify-end gap-2"):
            ui.button("Cancel", on_click=edit_dialog.close).props("flat")
            ui.button("Save", icon="save", on_click=lambda: save_edited_response()) \
                .props("color=primary unelevated")

    def ask_delete_chat(chat: dict):
        pending_delete["chat"] = chat
        title = chat.get("title") or "New chat"
        delete_label.text = f"Delete chat “{title}”? This can't be undone."
        delete_dialog.open()

    def delete_pending_chat():
        chat = pending_delete.get("chat")
        if not chat:
            return
        if generation_for_chat(chat.get("id")):
            ui.notify("Wait for generation to finish before deleting this chat.",
                      type="warning")
            return
        deleting_active = page["chat"] and page["chat"].get("id") == chat.get("id")
        store.delete_chat(chat["id"])
        pending_delete["chat"] = None
        if deleting_active:
            show_chat(first_chat(appstate.state.current_scenario))
        else:
            chat_list.refresh()

    def edit_last_response():
        chat = page["chat"]
        if not chat:
            return
        if generation_for_chat(chat["id"]):
            ui.notify("Wait for the current reply to finish.", type="warning")
            return
        index = _latest_assistant_response_index(chat.get("messages", []))
        if index is None:
            ui.notify("Only the latest assistant reply can be edited.", type="warning")
            return
        pending_edit["chat"] = chat
        pending_edit["index"] = index
        edit_box.value = chat["messages"][index].get("text", "")
        edit_dialog.open()

    def save_edited_response():
        chat = pending_edit.get("chat")
        index = pending_edit.get("index")
        text = (edit_box.value or "").strip()
        if not chat or index is None:
            edit_dialog.close()
            return
        if not text:
            ui.notify("Response text can't be empty.", type="warning")
            return
        if generation_for_chat(chat["id"]):
            ui.notify("Wait for the current reply to finish.", type="warning")
            return
        latest_index = _latest_assistant_response_index(chat.get("messages", []))
        if latest_index != index:
            ui.notify("Only the latest assistant reply can be edited.", type="warning")
            edit_dialog.close()
            return
        message = chat["messages"][index]
        message["text"] = text
        message.pop("finish_reason", None)
        message.pop("generation_error", None)
        message.pop("reasoning", None)
        store.save_chat(chat)
        pending_edit["chat"] = None
        pending_edit["index"] = None
        edit_dialog.close()
        if page["chat"] and page["chat"].get("id") == chat.get("id"):
            show_chat(chat)
        else:
            chat_list.refresh()

    def new_chat():
        scenario = appstate.state.current_scenario
        opening_text = scenario_details.get(scenario, {}).get("opening_text")
        show_chat(store.new_chat(scenario, appstate.state.current_model, opening_text))

    def on_scenario_change(name: str):
        appstate.state.current_scenario = name
        show_chat(first_chat(name))

    def start_assistant_reply(chat: dict, scenario: str | None):
        if page.get("inner") is None or page["inner"].is_deleted:
            render_messages()
        inner = page.get("inner")
        api = _api_messages(chat, scenario, scenario_details)
        model_name = engine.server.model or appstate.state.current_model
        assistant = {"role": "assistant", "name": scenario, "text": "", "model": model_name}
        chat["messages"].append(assistant)
        store.save_chat(chat)
        with inner:
            page["stream_view"] = _message(assistant, on_scenario_click=open_scenario)

        _start_generation(chat, api, assistant, dict(appstate.state.current_params))
        watch_generation(chat["id"])
        scroll_bottom()
        page["refresh_context"]()

    def send():
        text = _normalize_user_markdown(input_box.value or "")
        if not text:
            return
        if not engine.server.running:
            ui.notify("Load a model on the Model page first.", type="negative")
            return
        if page["chat"] and generation_for_chat(page["chat"]["id"]):
            # A second worker on the same chat would interleave its writes with
            # the first's and evict it from the registry.
            ui.notify("Wait for the current reply to finish.", type="warning")
            return
        if page["chat"] is None:
            new_chat()
        chat = page["chat"]
        scenario = appstate.state.current_scenario
        chat["messages"].append({"role": "user", "name": "You", "text": text})
        input_box.value = ""
        render_messages()
        scroll_bottom()
        start_assistant_reply(chat, scenario)

    def regenerate_last():
        chat = page["chat"]
        if not chat:
            return
        if not engine.server.running:
            ui.notify("Load a model on the Model page first.", type="negative")
            return
        if generation_for_chat(chat["id"]):
            ui.notify("Wait for the current reply to finish.", type="warning")
            return
        index = _latest_assistant_response_index(chat.get("messages", []))
        if index is None:
            ui.notify("Only the latest assistant reply can be regenerated.", type="warning")
            return
        scenario = chat.get("scenario") or appstate.state.current_scenario
        chat["messages"].pop(index)
        store.save_chat(chat)
        render_messages()
        scroll_bottom()
        start_assistant_reply(chat, scenario)

    @ui.refreshable
    def chat_list():
        chats = store.list_chats(appstate.state.current_scenario)
        if not chats:
            ui.label("No chats yet — start one.").classes("text-muted text-sm p-2")
            return
        with ui.list().classes("w-full tg-chat-list"):
            for c in chats:
                active = page["chat"] and page["chat"]["id"] == c["id"]
                item = ui.item(on_click=lambda cid=c["id"]: load_chat(cid)) \
                    .props("dense").classes("tg-chat-item w-full")
                if active:
                    item.classes("tg-active")
                with item, ui.item_section().classes("min-w-0"):
                    ui.label(c.get("title") or "New chat") \
                        .classes("font-medium text-sm ellipsis w-full")
                    ui.label(_rel_time(c.get("updated"))).classes("text-xs text-muted")
                with item, ui.item_section().props("side").classes("tg-chat-delete-section"):
                    ui.button(icon="delete", on_click=lambda chat=c: ask_delete_chat(chat)) \
                        .props("flat round dense size=sm text-color=white") \
                        .classes("tg-chat-delete") \
                        .tooltip("Delete chat")

    # ── Layout: sidebar + main ───────────────────────────────────────────────
    with ui.row().classes("w-full gap-4 no-wrap").style("height: calc(100vh - 7rem)"):
        with ui.column().classes("h-full w-64 shrink-0 gap-2 no-wrap"):
            ui.select(options=scenario_names, value=appstate.state.current_scenario,
                      label="Scenario",
                      on_change=lambda e: on_scenario_change(e.value)) \
                .props("filled").classes("w-full tg-field")
            ui.button("New chat", icon="add", on_click=new_chat) \
                .props("color=positive unelevated").classes("w-full")
            with ui.scroll_area().classes("flex-1 w-full min-h-0 tg-list-shell"):
                chat_list()

        with ui.column().classes("h-full flex-1 min-w-0 no-wrap gap-2"):
            with ui.scroll_area().classes("flex-1 w-full") as transcript_scroll:
                msgs_col = ui.column().classes("w-full")
            with ui.row().classes("w-full max-w-3xl mx-auto items-end gap-2 no-wrap"):
                input_box = ui.textarea(placeholder="Message…  (Ctrl+Enter to send)") \
                    .props("filled autogrow input-style=max-height:40vh") \
                    .classes("flex-1 tg-field")
                input_box.on("keydown.ctrl.enter", send)
                ui.button(icon="send", on_click=send) \
                    .props("color=primary unelevated").classes("h-14 w-14")

        with ui.column().classes("h-full w-56 shrink-0 gap-2 p-3 tg-list-shell"):
            ui.label("CONTEXT").classes("text-xs text-muted tracking-wide")
            context_counter = ui.badge("Context --") \
                .props("outline color=secondary") \
                .classes(
                    "min-h-8 w-full justify-center px-2 py-1 font-mono text-[11px] "
                    "whitespace-normal text-center leading-tight"
                )

    context_state = {"signature": None, "busy": False}

    def context_messages() -> list[dict]:
        chat = page["chat"] or {"messages": []}
        return _api_messages(chat, appstate.state.current_scenario, scenario_details,
                             draft=_normalize_user_markdown(input_box.value or ""))

    async def refresh_context_counter():
        if input_box.is_deleted or context_counter.is_deleted:
            context_timer.deactivate()
            return
        chat = page["chat"] or {"messages": []}
        last = chat["messages"][-1].get("text", "") if chat.get("messages") else ""
        total = _context_total()
        signature = (
            appstate.state.current_scenario,
            chat.get("id"),
            len(chat.get("messages", [])),
            last,
            input_box.value or "",
            total,
            engine.server.running,
            engine.server.model,
            engine.server.supports_system_role(),
        )
        if signature == context_state["signature"] or context_state["busy"]:
            return
        context_state["signature"] = signature
        context_state["busy"] = True
        try:
            used, exact = await run.io_bound(engine.server.count_chat_tokens, context_messages())
            context_counter.text = _context_label(used, total, exact)
        finally:
            context_state["busy"] = False

    def schedule_context_refresh(_=None):
        asyncio.create_task(refresh_context_counter())

    page["refresh_context"] = schedule_context_refresh
    input_box.on_value_change(schedule_context_refresh)
    context_timer = app.timer(1.0, refresh_context_counter, immediate=False)

    # Reopen the chat this browser left off on, so a reload lands back on the
    # one that may still be generating.
    _resumed = load_chat_state(appstate.state.current_chat_id)
    if not (_resumed and _resumed.get("scenario") == appstate.state.current_scenario):
        _resumed = first_chat(appstate.state.current_scenario)
    show_chat(_resumed)
