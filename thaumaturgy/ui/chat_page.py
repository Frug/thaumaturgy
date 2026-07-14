"""Chat page — conversation view with a per-character chat sidebar.

Left sidebar: character selector, "New chat", and the list of previous chats for
the selected character (chats are tied to characters, persisted as JSON). Main
area: the transcript + input. Send streams the assistant's reply live from the
llama-server engine using the selected character (context) + parameter set.
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


def _message(m: dict, on_char_click=None):
    """Render one message row; returns the markdown element (for live updates)."""
    is_user = m["role"] == "user"
    clickable = (not is_user) and on_char_click is not None
    with ui.row().classes("w-full gap-3 no-wrap items-start pb-4"):
        col = ui.column().classes("items-center gap-1 w-16 shrink-0")
        if clickable:
            col.classes("cursor-pointer hover:opacity-80")
            col.on("click", lambda: on_char_click(m["name"], m.get("model")))
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
    return md


async def _consume(gen_factory, on_delta):
    """Run a blocking generator in a thread; deliver its items to the UI loop."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    done = object()
    err: dict = {}

    def worker():
        try:
            for item in gen_factory():
                loop.call_soon_threadsafe(queue.put_nowait, item)
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller
            err["exc"] = exc
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, done)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = await queue.get()
        if item is done:
            break
        on_delta(item)
    if "exc" in err:
        raise err["exc"]


def _api_messages(chat: dict, character: str, char_details: dict) -> list[dict]:
    messages = []
    context = char_details.get(character, {}).get("context")
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
    characters = store.list_characters()
    char_names = [c["name"] for c in characters]
    char_details = {c["name"]: c for c in characters}
    if appstate.state.current_character not in char_names:
        appstate.state.current_character = char_names[0] if char_names else None
    page = {"chat": None}

    # ── Character info panel (slides in from the right) ──────────────────────
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

    def open_character(name: str, model: str | None = None):
        c = char_details.get(name, {"name": name, "context": "", "greeting": ""})
        detail.clear()
        with detail:
            with ui.column().classes("w-full gap-0 items-center"):
                ui.label("MODEL").classes("text-xs text-muted tracking-wide")
                model_name = model or "unknown"
                ui.badge(_truncate(model_name)).props("color=primary").classes(
                    "text-[10px] font-mono text-center break-all max-w-full").tooltip(model_name)
            ui.separator()
            with ui.avatar(color="secondary", size="88px").props("text-color=white"):
                ui.label((c["name"] or "?")[0].upper()).classes("text-3xl")
            ui.label(c["name"]).classes("text-lg font-semibold text-center")
            ui.separator()
            with ui.column().classes("w-full gap-1"):
                ui.label("CONTEXT").classes("text-xs text-muted tracking-wide")
                ui.markdown(c["context"] or "_None_").classes("text-sm leading-relaxed")
                ui.label("OPENING TEXT").classes("text-xs text-muted tracking-wide mt-3")
                ui.markdown(c["greeting"] or "_None_").classes("text-sm leading-relaxed")
        info_panel.classes(add="tg-open")
        backdrop.classes(add="tg-open")

    # ── Transcript rendering (container-based so we can stream into one msg) ──
    def render_messages():
        msgs_col.clear()
        page["inner"] = None
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
                for m in chat["messages"]:
                    _message(m, on_char_click=open_character)

    def scroll_bottom():
        transcript_scroll.scroll_to(percent=1.0)

    # ── Chat management ──────────────────────────────────────────────────────
    def load_chat(chat_id: str):
        page["chat"] = store.load_chat(chat_id)
        render_messages()
        chat_list.refresh()

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
        deleting_active = page["chat"] and page["chat"].get("id") == chat.get("id")
        store.delete_chat(chat["id"])
        pending_delete["chat"] = None
        if deleting_active:
            chats = store.list_chats(appstate.state.current_character)
            page["chat"] = chats[0] if chats else None
            render_messages()
        chat_list.refresh()

    def new_chat():
        char = appstate.state.current_character
        greeting = char_details.get(char, {}).get("greeting")
        page["chat"] = store.new_chat(char, appstate.state.current_model, greeting)
        render_messages()
        chat_list.refresh()

    def on_character_change(name: str):
        appstate.state.current_character = name
        chats = store.list_chats(name)
        page["chat"] = chats[0] if chats else None
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
        char = appstate.state.current_character
        chat["messages"].append({"role": "user", "name": "You", "text": text})
        input_box.value = ""
        render_messages()
        scroll_bottom()

        api = _api_messages(chat, char, char_details)  # up to and including the new user turn
        model_name = engine.server.model or appstate.state.current_model
        assistant = {"role": "assistant", "name": char, "text": "", "model": model_name}
        chat["messages"].append(assistant)
        with page["inner"]:
            md = _message(assistant, on_char_click=open_character)

        last = [0.0]

        def on_delta(delta: str):
            assistant["text"] += delta
            now = time.monotonic()
            if now - last[0] > 0.05:
                md.content = assistant["text"]
                scroll_bottom()
                last[0] = now

        try:
            await _consume(
                lambda: engine.server.stream_chat(api, appstate.state.current_params),
                on_delta)
        except Exception as exc:  # noqa: BLE001
            ui.notify(f"Generation error: {exc}", type="negative")
        md.content = assistant["text"] or "_(no output)_"
        scroll_bottom()
        store.save_chat(chat)
        chat_list.refresh()

    @ui.refreshable
    def chat_list():
        chats = store.list_chats(appstate.state.current_character)
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
                        .props("flat round dense size=sm") \
                        .classes("tg-chat-delete") \
                        .tooltip("Delete chat")

    # ── Layout: sidebar + main ───────────────────────────────────────────────
    with ui.row().classes("w-full gap-4 no-wrap").style("height: calc(100vh - 7rem)"):
        with ui.column().classes("h-full w-64 shrink-0 gap-2 no-wrap"):
            ui.select(options=char_names, value=appstate.state.current_character,
                      label="Character",
                      on_change=lambda e: on_character_change(e.value)) \
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

    _existing = store.list_chats(appstate.state.current_character)
    page["chat"] = _existing[0] if _existing else None
    render_messages()
    chat_list.refresh()
