"""Chat page — conversation view with a per-scenario chat sidebar.

Left sidebar: scenario selector, "New chat", and the list of previous chats for
the selected scenario (chats are tied to scenarios, persisted as JSON). Main
area: the transcript + input. Send streams the assistant's reply live from the
llama-server engine using the selected scenario + parameter set.
"""

import asyncio
import threading
import time

from nicegui import ui

from thaumaturgy import appstate, engine, store


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
    if reason == "length":
        return "Generation stopped because Max new tokens was reached."
    return f"Generation finished with reason: {reason}."


def _message(m: dict, on_scenario_click=None):
    """Render one message row; returns the markdown element (for live updates)."""
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
            md = ui.markdown(m.get("text") or "").classes("text-sm leading-relaxed break-words")
            warning = _finish_warning(m.get("finish_reason"))
            if warning:
                ui.badge(warning).props("color=warning text-color=dark") \
                    .classes("self-start text-xs mt-1")
    return md


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
        try:
            for event in engine.server.stream_chat(api, params):
                if event.get("type") == "finish":
                    assistant["finish_reason"] = event.get("reason")
                    continue
                delta = event.get("text", "")
                if not delta:
                    continue
                assistant["text"] += delta
                now = time.monotonic()
                if now - last_save > 0.5:
                    store.save_chat(chat)
                    last_save = now
        except Exception as exc:  # noqa: BLE001 - stored for the observing UI
            state["error"] = exc
        finally:
            state["done"] = True
            store.save_chat(chat)
            if appstate.state.generations.get(chat_id) is state:
                del appstate.state.generations[chat_id]

    threading.Thread(target=worker, daemon=True).start()
    return state


def _api_messages(chat: dict, scenario: str, scenario_details: dict) -> list[dict]:
    messages = []
    details = scenario_details.get(scenario, {})
    context = details.get("context")
    if context:
        messages.append({"role": "system", "content": context})
    for m in chat["messages"]:
        role = "user" if m["role"] == "user" else "assistant"
        messages.append({"role": role, "content": m.get("text", "")})
    return messages


def _truncate(text: str, max_len: int = 40) -> str:
    return text if len(text) <= max_len else f"{text[:max_len]}..."


def render():
    """Build the Chat page inside the current layout container."""
    scenarios = store.list_scenarios()
    scenario_names = [s["name"] for s in scenarios]
    scenario_details = {s["name"]: s for s in scenarios}
    if appstate.state.current_scenario not in scenario_names:
        appstate.state.current_scenario = scenario_names[0] if scenario_names else None
    page = {"chat": None}

    def generation_for_chat(chat_id: str | None) -> dict | None:
        return appstate.state.generations.get(chat_id or "")

    def load_chat_state(chat_id: str | None) -> dict | None:
        generation = generation_for_chat(chat_id)
        if generation:
            return generation["chat"]
        return store.load_chat(chat_id) if chat_id else None

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
    def render_messages():
        msgs_col.clear()
        page["inner"] = None
        page["stream_md"] = None
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
                for i, m in enumerate(chat["messages"]):
                    md = _message(m, on_scenario_click=open_scenario)
                    if generation and i == generation["assistant_index"]:
                        page["stream_md"] = md

    def scroll_bottom():
        transcript_scroll.scroll_to(percent=1.0)

    async def observe_generation(generation: dict):
        last = [0.0]
        while not generation["done"]:
            await asyncio.sleep(0.05)
            if msgs_col.is_deleted or transcript_scroll.is_deleted:
                return
            if not page["chat"] or page["chat"].get("id") != generation["chat_id"]:
                return
            md = page.get("stream_md")
            if md is None or md.is_deleted:
                return
            now = time.monotonic()
            if now - last[0] > 0.05:
                md.content = generation["assistant"].get("text", "")
                scroll_bottom()
                last[0] = now

        if msgs_col.is_deleted or transcript_scroll.is_deleted:
            return
        if not page["chat"] or page["chat"].get("id") != generation["chat_id"]:
            return
        if generation["error"]:
            ui.notify(f"Generation error: {generation['error']}", type="negative")
        md = page.get("stream_md")
        if md is not None and not md.is_deleted:
            md.content = generation["assistant"].get("text") or "_(no output)_"
        if generation["assistant"].get("finish_reason") and generation["assistant"]["finish_reason"] != "stop":
            render_messages()
        scroll_bottom()
        chat_list.refresh()

    # ── Chat management ──────────────────────────────────────────────────────
    def load_chat(chat_id: str):
        page["chat"] = load_chat_state(chat_id)
        appstate.state.current_chat_id = chat_id if page["chat"] else None
        render_messages()
        chat_list.refresh()
        generation = generation_for_chat(chat_id)
        if generation:
            asyncio.create_task(observe_generation(generation))

    pending_delete = {"chat": None}

    with ui.dialog() as delete_dialog, ui.card().classes("p-5 gap-3").style("width:420px;max-width:92vw"):
        delete_label = ui.label().classes("text-sm leading-relaxed")
        with ui.row().classes("w-full justify-end gap-2"):
            ui.button("Cancel", on_click=delete_dialog.close).props("flat")
            ui.button("Delete", icon="delete",
                      on_click=lambda: (delete_dialog.close(), delete_pending_chat())) \
                .props("color=negative unelevated")

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
            chats = store.list_chats(appstate.state.current_scenario)
            page["chat"] = chats[0] if chats else None
            appstate.state.current_chat_id = page["chat"]["id"] if page["chat"] else None
            render_messages()
        chat_list.refresh()

    def new_chat():
        scenario = appstate.state.current_scenario
        opening_text = scenario_details.get(scenario, {}).get("opening_text")
        page["chat"] = store.new_chat(scenario, appstate.state.current_model, opening_text)
        appstate.state.current_chat_id = page["chat"]["id"]
        render_messages()
        chat_list.refresh()

    def on_scenario_change(name: str):
        appstate.state.current_scenario = name
        chats = store.list_chats(name)
        page["chat"] = chats[0] if chats else None
        appstate.state.current_chat_id = page["chat"]["id"] if page["chat"] else None
        render_messages()
        chat_list.refresh()

    async def send():
        text = (input_box.value or "").strip()
        if not text:
            return
        if not engine.server.running:
            ui.notify("Load a model on the Model page first.", type="negative")
            return
        if page["chat"] is None:
            new_chat()
        chat = page["chat"]
        scenario = appstate.state.current_scenario
        chat["messages"].append({"role": "user", "name": "You", "text": text})
        input_box.value = ""
        render_messages()
        scroll_bottom()

        api = _api_messages(chat, scenario, scenario_details)
        model_name = engine.server.model or appstate.state.current_model
        assistant = {"role": "assistant", "name": scenario, "text": "", "model": model_name}
        chat["messages"].append(assistant)
        store.save_chat(chat)
        with page["inner"]:
            md = _message(assistant, on_scenario_click=open_scenario)
            page["stream_md"] = md

        generation = _start_generation(chat, api, assistant,
                                       dict(appstate.state.current_params))
        await observe_generation(generation)

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
            with ui.column().classes("flex-1 w-full min-h-0 overflow-y-auto rounded-lg gap-0 p-0") \
                    .style("background: rgba(52,97,140,0.06)"):
                chat_list()

        with ui.column().classes("h-full flex-1 no-wrap gap-2"):
            with ui.scroll_area().classes("flex-1 w-full") as transcript_scroll:
                msgs_col = ui.column().classes("w-full")
            with ui.row().classes("w-full max-w-3xl mx-auto items-end gap-2 no-wrap"):
                input_box = ui.textarea(placeholder="Message…  (Ctrl+Enter to send)") \
                    .props("filled autogrow input-style=max-height:40vh") \
                    .classes("flex-1 tg-field")
                input_box.on("keydown.ctrl.enter", send)
                ui.button(icon="send", on_click=send) \
                    .props("color=primary unelevated").classes("h-14 w-14")

    _existing = store.list_chats(appstate.state.current_scenario)
    _current_chat = load_chat_state(appstate.state.current_chat_id)
    if _current_chat and _current_chat.get("scenario") == appstate.state.current_scenario:
        page["chat"] = _current_chat
    else:
        page["chat"] = _existing[0] if _existing else None
        appstate.state.current_chat_id = page["chat"]["id"] if page["chat"] else None
    render_messages()
    chat_list.refresh()
    generation = generation_for_chat(page["chat"]["id"] if page["chat"] else None)
    if generation:
        asyncio.create_task(observe_generation(generation))
