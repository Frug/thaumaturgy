"""Chat page: conversation view with a per-scenario chat sidebar.

Rendering only. The chat service owns the conversation, its scenario, and any
reply in flight; this module draws it and collects input.
"""

import asyncio
import re
import time

from nicegui import app, run, ui

from thaumaturgy import appstate, engine
from thaumaturgy.chat import Message, Step, chat
from thaumaturgy.ui.outcomes import notify

# Each streamed update re-parses the whole message as markdown, so cap the
# re-render rate and prefer to land it on a newline boundary.
_STREAM_MIN_INTERVAL = 0.2
_STREAM_MAX_INTERVAL = 0.4


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


def _truncate(text: str, max_len: int = 40) -> str:
    return text if len(text) <= max_len else f"{text[:max_len]}..."


def _normalize_user_markdown(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    return re.sub(r"(?<!\n)\n(?!\n)", "\n\n", text)


def _soften_indent(text: str) -> str:
    """Drop per-line leading whitespace outside fenced code.

    Models indent their scratchpad bullets, which markdown renders as code
    blocks; flattening keeps them as lists and prose.
    """
    if "\n" not in text and not text[:1].isspace():
        return text
    out, fenced = [], False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            out.append(line)
        else:
            out.append(line if fenced else line.lstrip())
    return "\n".join(out)


def _context_label(used: int | None, total: int | None, exact: bool = True) -> str:
    if used is None:
        return "Context --"
    prefix = "" if exact else "~"
    if total:
        pct = min(999, round((used / total) * 100))
        return f"Context {prefix}{used:,} / {total:,} ({pct}%)"
    return f"Context {prefix}{used:,}"


def _avatar(m: Message):
    with ui.avatar(color="primary" if m.is_user else "secondary") \
            .props("text-color=white"):
        ui.label((m.name or "?")[0].upper())


class _MessageView:
    """Live handles into one rendered message, so a stream can update it in place.

    The Thinking pane is built up front and hidden because the bubble's slot is
    closed by the time reasoning arrives; an observer can't add elements then.
    """

    def __init__(self, text_md, reasoning_box=None, reasoning_md=None):
        self.text_md = text_md
        self.reasoning_box = reasoning_box
        self.reasoning_md = reasoning_md

    @property
    def is_deleted(self) -> bool:
        return self.text_md.is_deleted

    def update(self, text: str, reasoning: str) -> None:
        self.text_md.content = _soften_indent(text)
        if self.reasoning_box is None:
            return
        self.reasoning_md.content = _soften_indent(reasoning)
        self.reasoning_box.set_visibility(bool(reasoning.strip()))


def _message(m: Message, on_scenario_click=None) -> _MessageView:
    """Render one message row; returns handles to it (for live updates)."""
    clickable = (not m.is_user) and on_scenario_click is not None
    with ui.row().classes("w-full gap-3 no-wrap items-start pb-4"):
        col = ui.column().classes("items-center gap-1 w-16 shrink-0")
        if clickable:
            col.classes("cursor-pointer hover:opacity-80")
            col.on("click", lambda: on_scenario_click(m.model))
        with col:
            _avatar(m)
            ui.label(m.name or "").classes(
                "text-xs text-center leading-tight "
                + ("text-primary" if clickable else "text-muted"))
        bubble = ui.column().classes("flex-1 min-w-0 gap-1 p-3 rounded-xl")
        if m.is_user:
            bubble.style("background: rgba(52,97,140,0.10)")
        with bubble:
            text, reasoning = m.display()
            md = ui.markdown(_soften_indent(text)).classes(
                "text-sm leading-relaxed break-words")
            box = reasoning_md = None
            if not m.is_user:
                box = ui.expansion("Thinking", icon="psychology").classes("w-full")
                with box:
                    reasoning_md = ui.markdown(_soften_indent(reasoning)).classes(
                        "text-xs leading-relaxed break-words text-muted")
                box.set_visibility(bool(reasoning))
            warning = m.warning()
            if warning:
                ui.badge(warning).props("color=warning text-color=dark") \
                    .classes("self-start text-xs mt-1")
    return _MessageView(md, box, reasoning_md)


def render():
    """Build the Chat page inside the current layout container."""
    scenarios = chat.scenarios()
    names = [s.name for s in scenarios]
    if chat.scenario_name not in names:
        chat.scenario_name = names[0] if names else None
        appstate.state.current_scenario = chat.scenario_name
    page: dict = {"inner": None, "stream_view": None, "observed": None,
                  "refresh_context": lambda: None}

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
        scenario = chat.scenario()
        detail.clear()
        with detail:
            with ui.column().classes("w-full gap-0 items-center"):
                ui.label("MODEL").classes("text-xs text-muted tracking-wide")
                model_name = model or "unknown"
                ui.badge(_truncate(model_name)).props("color=primary").classes(
                    "text-[10px] font-mono text-center break-all max-w-full") \
                    .tooltip(model_name)
            ui.separator()
            with ui.avatar(color="secondary", size="88px").props("text-color=white"):
                ui.label(((scenario.name if scenario else "") or "?")[0].upper()) \
                    .classes("text-3xl")
            ui.label(scenario.name if scenario else "").classes(
                "text-lg font-semibold text-center")
            ui.separator()
            with ui.column().classes("w-full gap-1"):
                ui.label("SCENARIO CONTEXT").classes("text-xs text-muted tracking-wide")
                ui.markdown((scenario.context if scenario else "") or "_None_") \
                    .classes("text-sm leading-relaxed")
                ui.label("OPENING TEXT").classes("text-xs text-muted tracking-wide mt-3")
                ui.markdown((scenario.opening_text if scenario else "") or "_None_") \
                    .classes("text-sm leading-relaxed")
        info_panel.classes(add="tg-open")
        backdrop.classes(add="tg-open")

    # ── Transcript ───────────────────────────────────────────────────────────
    def render_reply_actions():
        with ui.row().classes("w-full gap-2 no-wrap items-start pb-2"):
            ui.element("div").classes("w-16 shrink-0")
            ui.button("Regenerate", icon="refresh", on_click=regenerate_last) \
                .props("flat dense color=secondary").classes("text-xs")
            ui.button("Edit", icon="edit", on_click=edit_last_response) \
                .props("flat dense color=secondary").classes("text-xs")

    def render_messages():
        msgs_col.clear()
        page["inner"] = None
        page["stream_view"] = None
        with msgs_col:
            if chat.chat is None:
                with ui.column().classes("w-full h-full items-center justify-center gap-2"):
                    ui.icon("forum").classes("text-5xl text-muted")
                    ui.label("Start a new chat.").classes("text-muted")
                return
            inner = ui.column().classes("w-full max-w-3xl mx-auto gap-2")
            page["inner"] = inner
            with inner:
                run_ = chat.run
                streaming = chat.busy()
                regenerate_index = (None if streaming
                                    else chat.chat.latest_assistant_index())
                for i, m in enumerate(chat.chat.messages):
                    view = _message(m, on_scenario_click=open_scenario)
                    if streaming and run_ is not None and i == run_.index:
                        page["stream_view"] = view
                    if i == regenerate_index:
                        render_reply_actions()

    def scroll_bottom():
        transcript_scroll.scroll_to(percent=1.0)

    async def scroll_bottom_after_render():
        await asyncio.sleep(0.05)
        if not transcript_scroll.is_deleted:
            scroll_bottom()

    def still_showing(run_) -> bool:
        """True while this page is still showing the chat this reply feeds."""
        if msgs_col.is_deleted or transcript_scroll.is_deleted:
            return False
        return chat.chat is not None and chat.chat.id == run_.chat_id

    async def observe(run_):
        """Mirror a running reply into the transcript until it finishes."""
        last = None
        last_render = 0.0
        last_newlines = 0
        try:
            while not run_.done:
                await asyncio.sleep(0.1)
                if not still_showing(run_):
                    return
                view = page["stream_view"]
                if view is None or view.is_deleted:
                    return
                current = run_.snapshot
                if current == last:
                    continue
                now = time.monotonic()
                elapsed = now - last_render
                newlines = current[0].count("\n") + current[1].count("\n")
                if elapsed >= _STREAM_MAX_INTERVAL or (
                        elapsed >= _STREAM_MIN_INTERVAL and newlines > last_newlines):
                    view.update(*current)
                    scroll_bottom()
                    last = current
                    last_render = now
                    last_newlines = newlines

            if not still_showing(run_):
                return
            outcome = chat.complete_run(run_)
            message = run_.message
            view = page["stream_view"]
            if view is not None and not view.is_deleted:
                text, reasoning = message.display()
                view.update(text or "_(no output)_", reasoning)
            if message.warning():
                render_messages()  # re-render to hang the warning badge off the bubble
            elif chat.chat.latest_assistant_index() == run_.index:
                with page["inner"]:
                    render_reply_actions()
            scroll_bottom()
            chat_list.refresh()
            if outcome.step is Step.ERROR:
                # Runs as a bare task, which has no slot stack of its own;
                # ui.notify needs one to find the client.
                with msgs_col:
                    notify(outcome)
        finally:
            if page["observed"] is run_:
                page["observed"] = None

    def watch():
        """Observe the reply feeding the open chat, if there is one."""
        run_ = chat.run
        if run_ is not None and not run_.done and page["observed"] is not run_:
            page["observed"] = run_  # at most one observer at a time
            asyncio.create_task(observe(run_))

    # ── Chat management ──────────────────────────────────────────────────────
    def show_current():
        render_messages()
        chat_list.refresh()
        watch()
        page["refresh_context"]()
        asyncio.create_task(scroll_bottom_after_render())

    def apply(outcome):
        notify(outcome)
        show_current()

    def load_chat(chat_id: str):
        apply(chat.open(chat_id))

    pending_delete = {"chat_id": None}
    pending_edit = {"chat_id": None}

    with ui.dialog() as delete_dialog, ui.card().classes("p-5 gap-3") \
            .style("width:420px;max-width:92vw"):
        delete_label = ui.label().classes("text-sm leading-relaxed")
        with ui.row().classes("w-full justify-end gap-2"):
            ui.button("Cancel", on_click=delete_dialog.close).props("flat")
            ui.button("Delete", icon="delete",
                      on_click=lambda: (delete_dialog.close(), delete_pending_chat())) \
                .props("color=negative unelevated")

    with ui.dialog() as edit_dialog, ui.card().classes("p-5 gap-3") \
            .style("width:720px;max-width:92vw"):
        ui.label("Edit Response").classes("text-lg font-semibold")
        edit_box = ui.textarea().props("filled autogrow input-style=max-height:60vh") \
            .classes("w-full tg-field")
        with ui.row().classes("w-full justify-end gap-2"):
            ui.button("Cancel", on_click=edit_dialog.close).props("flat")
            ui.button("Save", icon="save", on_click=lambda: save_edited_response()) \
                .props("color=primary unelevated")

    def ask_delete_chat(raw: dict):
        pending_delete["chat_id"] = raw["id"]
        delete_label.text = f"Delete chat “{raw.get('title') or 'New chat'}”? " \
                            "This can't be undone."
        delete_dialog.open()

    def delete_pending_chat():
        chat_id = pending_delete["chat_id"]
        if chat_id:
            apply(chat.delete(chat_id))
        pending_delete["chat_id"] = None

    def edit_last_response():
        message = chat.editable_reply()
        if message is None or chat.busy():
            notify(chat.edit_last(""))  # reuses the service's own guard messages
            return
        pending_edit["chat_id"] = chat.chat.id
        edit_box.value = message.text
        edit_dialog.open()

    def save_edited_response():
        if pending_edit["chat_id"] != (chat.chat.id if chat.chat else None):
            edit_dialog.close()
            return
        outcome = chat.edit_last(edit_box.value or "")
        if outcome.step is Step.BLOCKED:
            notify(outcome)
            return
        edit_dialog.close()
        pending_edit["chat_id"] = None
        apply(outcome)

    def new_chat():
        apply(chat.new_chat())

    def on_scenario_change(name: str):
        apply(chat.select_scenario(name))

    def send():
        text = _normalize_user_markdown(input_box.value or "")
        if not text:
            return
        input_box.value = ""
        outcome = chat.send(text)
        if outcome.step is Step.BLOCKED:
            notify(outcome)
            return
        apply(outcome)

    def regenerate_last():
        apply(chat.regenerate())

    @ui.refreshable
    def chat_list():
        chats = chat.list_chats(chat.scenario_name)
        if not chats:
            ui.label("No chats yet — start one.").classes("text-muted text-sm p-2")
            return
        with ui.list().classes("w-full tg-chat-list"):
            for raw in chats:
                active = chat.chat is not None and chat.chat.id == raw["id"]
                item = ui.item(on_click=lambda cid=raw["id"]: load_chat(cid)) \
                    .props("dense").classes("tg-chat-item w-full")
                if active:
                    item.classes("tg-active")
                with item, ui.item_section().classes("min-w-0"):
                    ui.label(raw.get("title") or "New chat") \
                        .classes("font-medium text-sm ellipsis w-full")
                    ui.label(_rel_time(raw.get("updated"))).classes("text-xs text-muted")
                with item, ui.item_section().props("side").classes("tg-chat-delete-section"):
                    ui.button(icon="delete", on_click=lambda r=raw: ask_delete_chat(r)) \
                        .props("flat round dense size=sm text-color=white") \
                        .classes("tg-chat-delete").tooltip("Delete chat")

    # ── Layout: sidebar + main ───────────────────────────────────────────────
    with ui.row().classes("w-full gap-4 no-wrap").style("height: calc(100vh - 7rem)"):
        with ui.column().classes("h-full w-64 shrink-0 gap-2 no-wrap"):
            ui.select(options=names, value=chat.scenario_name, label="Scenario",
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
                .classes("min-h-8 w-full justify-center px-2 py-1 font-mono "
                         "text-[11px] whitespace-normal text-center leading-tight")

    context_state = {"signature": None, "busy": False}

    async def refresh_context_counter():
        if input_box.is_deleted or context_counter.is_deleted:
            context_timer.deactivate()
            return
        messages = chat.chat.messages if chat.chat else []
        last = messages[-1].text if messages else ""
        total = chat.context_total()
        signature = (chat.scenario_name, chat.chat.id if chat.chat else None,
                     len(messages), last, input_box.value or "", total,
                     engine.server.running, engine.server.model,
                     engine.server.supports_system_role())
        if signature == context_state["signature"] or context_state["busy"]:
            return
        context_state["signature"] = signature
        context_state["busy"] = True
        try:
            draft = _normalize_user_markdown(input_box.value or "")
            used, exact = await run.io_bound(engine.server.count_chat_tokens,
                                             chat.context_messages(draft))
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
    if not (chat.chat and chat.chat.scenario == chat.scenario_name):
        chat.open_first(chat.scenario_name)
    show_current()
