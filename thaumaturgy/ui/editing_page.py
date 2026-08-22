"""Editing page: walk a long document past the model one span at a time.

Rendering only. The service owns the job, the run, and the rules; this module
shows what it reports and collects decisions.
"""

import asyncio

from nicegui import app, context, ui

from thaumaturgy import appstate, engine, store
from thaumaturgy.editing import (EditingService, Instructions, Settings, Status,
                                 Step, editing_runtime)
from thaumaturgy.editing.spans import est_tokens
from thaumaturgy.lang import en
from thaumaturgy.ui.outcomes import notify

FLAG_TEXT = {
    "truncated": "Reply hit the token cap",
    "empty": "Model returned nothing",
    "length-ratio": "Length differs sharply from the original",
    "context-bleed": "Output ran past the span markers",
    "lost-break": "A paragraph break inside the span was dropped",
    "invented": "Contains prose that isn't in the original",
    "stops-short": "Stops partway — the end of the span is missing",
    "starts-late": "Skips the opening — the start of the span is missing",
    "error": "Generation failed",
}

ROLE_COLOR = {"system": "purple", "user": "primary", "assistant": "teal"}

_DECIDED = (str(Status.ACCEPTED), str(Status.ORIGINAL))

HELP = {
    "title": "Name for this job in the sidebar. Defaults to the uploaded "
             "file's name.",
    "document": "The text to edit. Paste it here, or upload a .txt/.md file "
                "above. It is copied into the job, so editing never touches "
                "your original file.",
    "saved": "Load a saved set of instructions and prompt wrapper. Picking one "
             "replaces every prompt field below; Save as… stores the current "
             "ones under a name.",
    "system": "Your instructions to the model, sent as the system prompt with "
              "every passage. This is the main thing to tune — say what may "
              "change and, just as importantly, what must come back untouched.",
    "max_new": "Cap on the model's reply for one passage. It also sets how big "
               "a passage is: passage size = this minus the response buffer. "
               "Models get unreliable at reproducing text much past ~2,000 "
               "words, and well before that.",
    "buffer": "Headroom between the passage size and the reply cap, so a "
              "passage that grows a little under editing still fits. If spans "
              "keep hitting the cap and splitting, raise this.",
    "overlap": "How much surrounding text the model sees either side of the "
               "passage, as a share of the free context. It buys consistent "
               "wording across joins, but it is the single biggest cause of a "
               "model drifting out of its passage — 0 is the most faithful. "
               "Because it is a share of the loaded context, the same "
               "percentage means far more text at a large context size.",
    "temperature": "Sampling randomness. Editing wants near-deterministic "
                   "output, so keep this low; 0.2 or below.",
    "instruction": "Wrapped around the passage in the final turn. "
                   "{first_words} and {last_words} quote the passage's own "
                   "ends back, which is what stops the model returning only "
                   "part of it.",
    "framing": "Introduces the surrounding text when overlap is above 0. "
               "{before} and {after} are where that text lands.",
    "prime": "Inserts a reply written as though the model said it, agreeing to "
             "treat the context as reference only. Models follow their own "
             "prior turns more readily than instructions, so this measurably "
             "helps — but it can also override your system prompt, which is "
             "why it is off by default.",
    "primed_text": "The words put in the model's mouth. Only sent when the "
                   "switch above is on, and only when overlap is above 0.",
    "deletions": "Turn on when you're asking the model to strip content, so a "
                 "shrinking passage isn't treated as a fault. Invented prose "
                 "and truncation are still flagged.",
    "auto_accept": "Accept passages that pass every check and move straight on, "
                   "stopping only on flagged ones. The checks are mechanical — "
                   "they catch a mangled passage, not a bad edit — so leave "
                   "this off until you've watched a few spans.",
}


def _explain(field, text: str):
    """Hang a hoverable ? inside a field, on the right."""
    with field.add_slot("append"):
        ui.icon("help_outline").classes("text-sm text-muted cursor-help").tooltip(text)
    return field


def _switch(label: str, text: str):
    """A switch with the same ? beside it; QToggle has no append slot."""
    with ui.row().classes("items-center gap-1 no-wrap"):
        toggle = ui.switch(label)
        ui.icon("help_outline").classes("text-sm text-muted cursor-help").tooltip(text)
    return toggle


def _banner(text: str, color: str = "warning"):
    """A full-width badge across the top of the page."""
    props = f"color={color}" if color == "negative" else f"color={color} text-color=dark"
    return ui.badge(text).props(props).classes("text-xs p-2 whitespace-normal")


def _stored_percent(raw: dict) -> int:
    """Progress of a job on disk, without loading its whole document."""
    spans = raw.get("spans") or []
    if not spans:
        return 0
    done = sum(s["end"] - s["start"] for s in spans if s.get("status") in _DECIDED)
    total = spans[-1]["end"] - spans[0]["start"]
    return round(done / total * 100) if total else 0


async def render():
    """Build the Editing page inside the current layout container."""
    await context.client.connected()
    tab = app.storage.tab
    user = app.storage.user
    editor = EditingService(
        editing_runtime,
        model_name=tab.get("selected_model") or user.get("selected_model"),
    )
    remembered = tab.get("editing_job_id") or user.get("editing_job_id")
    available = {raw["id"] for raw in store.list_jobs()}
    if remembered in available:
        editor.open(remembered)
    page: dict = {"nudge": "", "stream_box": None, "watching": False}

    # ── Outcome rendering ───────────────────────────────────────────────────
    def show(outcome) -> None:
        """Render whatever the service just did, and watch any run it started."""
        notify(outcome)
        job_list.refresh()
        show_panels()
        watch()

    def watch() -> None:
        if editor.running and not page["watching"]:
            asyncio.create_task(observe())

    async def observe() -> None:
        """Mirror runs into the page until the service stops starting them.

        Loops rather than handling one run: the service chains spans itself on
        split-retry and auto-accept.
        """
        page["watching"] = True
        try:
            while editor.running:
                run = editor.run
                last = None
                while not run.done:
                    await asyncio.sleep(0.15)
                    box = page["stream_box"]
                    if box is None or box.is_deleted:
                        return  # page gone; the run survives on the service
                    if run.text != last:
                        last = run.text
                        box.set_content(f"```\n{run.text}\n```")
                notify(editor.complete_run(run))
                job_list.refresh()
                review.refresh()
        finally:
            page["watching"] = False

    # ── Warnings ────────────────────────────────────────────────────────────
    @ui.refreshable
    def warnings():
        if not engine.server.running:
            _banner("No model loaded — load one on the Model page.", "negative")
            return
        if engine.server.thinking_enabled() and engine.server.reasoning_budget < 0:
            _banner("This model thinks with an unrestricted budget, so a span "
                    "rewrite is capped only by the context window. Set reasoning "
                    "off or a budget on the Model page.")
        if appstate.state.generations:
            _banner("A chat generation is running — editing shares the one server.")

    # ── Job list ────────────────────────────────────────────────────────────
    @ui.refreshable
    def job_list():
        jobs = store.list_jobs()
        if not jobs:
            ui.label("No documents yet — click New.").classes("text-muted text-sm p-3")
            return
        with ui.list().classes("w-full"):
            for raw in jobs:
                active = editor.job is not None and editor.job.id == raw["id"]
                item = ui.item(on_click=lambda jid=raw["id"]: open_job(jid)) \
                    .props("dense").classes("tg-nav-item w-full")
                if active:
                    item.classes("tg-active")
                with item, ui.item_section().classes("min-w-0"):
                    ui.label(raw.get("title") or "Untitled") \
                        .classes("font-medium text-sm ellipsis w-full")
                    ui.label(f"{_stored_percent(raw)}% edited").classes("text-xs text-muted")
                with item, ui.item_section().props("side"):
                    ui.button(icon="delete",
                              on_click=lambda jid=raw["id"]: remove_job(jid)) \
                        .props("flat round dense size=sm").tooltip("Delete document")

    def remove_job(job_id: str):
        if editor.occupied(job_id):
            ui.notify(en.SPAN_BUSY, type="warning")
            return
        store.delete_job(job_id)
        editing_runtime.forget(job_id)
        if editor.job is not None and editor.job.id == job_id:
            editor.close()
        job_list.refresh()
        show_panels()

    # ── Intake ──────────────────────────────────────────────────────────────
    async def take_upload(e):
        # Broad catch on purpose: NiceGUI logs a handler exception server-side
        # and still tells the browser the upload succeeded, so anything escaping
        # here looks to the user like the file silently did nothing.
        try:
            text = await e.file.text()
        except Exception as exc:  # noqa: BLE001
            ui.notify(f"Could not read that file: {exc}", type="negative")
            return
        doc_box.value = text
        if not title_box.value:
            title_box.value = e.file.name
        ui.notify(f"Loaded {e.file.name} ({len(text):,} characters)")

    def reject_upload(_):
        # Quasar filters on `accept` and `max-files` before uploading; without
        # this the file just vanishes with no indication of why.
        ui.notify("Not accepted — one .txt or .md file at a time. Clear the "
                  "current file to choose another.", type="warning")

    def form_settings() -> Settings:
        return Settings.from_dict({
            "max_new_tokens": max_new.value,
            "temperature": temperature.value,
            "response_buffer": buffer_box.value,
            "overlap_pct": (overlap.value or 0) / 100.0,
            "auto_accept_clean": auto_accept.value,
            "allow_deletions": allow_deletions.value,
        })

    def form_instructions() -> Instructions:
        return Instructions.from_dict(current_instructions())

    def create_job():
        text = doc_box.value or ""
        if not text.strip():
            ui.notify("Paste or upload a document first.", type="warning")
            return
        show(editor.create(title_box.value or "Untitled document", text,
                           form_instructions(), form_settings()))

    def open_job(job_id: str):
        if editor.running:
            ui.notify(en.SPAN_BUSY, type="warning")
            return
        show(editor.open(job_id))

    # ── Decisions ───────────────────────────────────────────────────────────
    def accept():
        show(editor.accept())

    def keep_original():
        show(editor.keep_original())

    def retry():
        show(editor.retry(page["nudge"]))

    def stop():
        editor.stop()

    def open_edit():
        span = editor.span
        if span is None or editor.running:
            return
        edit_area.value = span.rewritten
        edit_dialog.open()

    def save_edit():
        outcome = editor.accept(edit_area.value or "")
        if outcome.step is not Step.BLOCKED:
            edit_dialog.close()
        show(outcome)

    def restart_job():
        """Back to the intake form with this job's settings, to run it again.

        Non-destructive: pressing Start editing makes a new job from the
        original document, and the one being restarted keeps its decisions.
        """
        job = editor.job
        if job is None:
            return
        if editor.running:
            ui.notify("Stop the current span first.", type="warning")
            return
        s, instr = job.settings, job.instructions
        title_box.value = job.title
        doc_box.value = job.source_text
        max_new.value = s.max_new_tokens
        buffer_box.value = s.response_buffer
        overlap.value = round(s.overlap_pct * 100)
        temperature.value = s.temperature
        auto_accept.value = s.auto_accept_clean
        allow_deletions.value = s.allow_deletions
        apply_instructions(instr.to_dict())
        editor.close()
        job_list.refresh()
        show_panels()
        ui.notify("Loaded this job's document and settings. Adjust anything, "
                  "then press Start editing — the old job is left as it is.")

    def export():
        job = editor.job
        if job is None:
            return
        ui.download.content(editor.export_text(), f"{job.title or 'document'}-edited.txt")

    with ui.dialog() as edit_dialog, ui.card().classes("p-5 gap-3") \
            .style("width:820px;max-width:94vw"):
        ui.label("Edit rewrite").classes("text-lg font-semibold")
        edit_area = ui.textarea().props("filled autogrow input-style=max-height:60vh") \
            .classes("w-full tg-field")
        with ui.row().classes("w-full justify-end gap-2"):
            ui.button("Cancel", on_click=edit_dialog.close).props("flat")
            ui.button("Save & accept", icon="check", on_click=save_edit) \
                .props("color=primary unelevated")

    # ── Saved instruction sets ──────────────────────────────────────────────
    # The prompt text is the part worth carrying between documents; the token
    # sizes belong to the document and the model, so they stay with the job.
    picking = {"suppress": False}

    def current_instructions() -> dict:
        return {
            "system_prompt": system_box.value or "",
            "passage_instruction": instruction_box.value or "",
            "context_framing": framing_box.value or "",
            "primed_reply": primed_box.value or "",
            "prime_reply": bool(prime_reply.value),
        }

    def apply_instructions(vals: dict) -> None:
        v = Instructions.from_dict(vals)
        system_box.value = v.system_prompt
        instruction_box.value = v.passage_instruction
        framing_box.value = v.context_framing
        primed_box.value = v.primed_reply
        prime_reply.value = v.prime_reply

    def refresh_saved(select_name: str | None = None) -> None:
        sets = store.load_edit_prompts()
        if not sets:
            sets = {"Copy edit": Instructions().to_dict()}
            store.save_edit_prompts(sets)
        names = sorted(sets)
        # Rebuilding the list moves `value`, which fires the change handler and
        # would overwrite the fields. Only a real pick should load.
        picking["suppress"] = True
        try:
            saved_select.options = names
            if select_name in names:
                saved_select.value = select_name
            elif saved_select.value not in names:
                # Left unselected rather than defaulted, so the box never names a
                # set the fields below don't actually contain.
                saved_select.value = None
            saved_select.update()
        finally:
            picking["suppress"] = False

    def load_saved(name: str | None) -> None:
        if picking["suppress"] or not name:
            return
        sets = store.load_edit_prompts()
        if name not in sets:
            return
        apply_instructions(sets[name])
        ui.notify(f"Loaded “{name}”.")

    def open_save() -> None:
        save_name.value = saved_select.value or ""
        save_dialog.open()

    def do_save() -> None:
        name = (save_name.value or "").strip()
        if not name:
            ui.notify("Give the instruction set a name.", type="warning")
            return
        sets = store.load_edit_prompts()
        replacing = name in sets
        sets[name] = current_instructions()
        store.save_edit_prompts(sets)
        save_dialog.close()
        refresh_saved(name)
        ui.notify(f"{'Replaced' if replacing else 'Saved'} “{name}”.")

    def delete_saved() -> None:
        sets = store.load_edit_prompts()
        name = saved_select.value
        if name not in sets:
            return
        del sets[name]
        store.save_edit_prompts(sets)
        refresh_saved()
        ui.notify(f"Deleted “{name}”.")

    with ui.dialog() as save_dialog, ui.card().classes("p-5 gap-3") \
            .style("width:460px;max-width:92vw"):
        ui.label("Save editing instructions").classes("text-lg font-semibold")
        ui.label("Stores the instructions and the prompt wrapper, ready to load "
                 "onto the next document. An existing name is overwritten.") \
            .classes("text-xs text-muted")
        save_name = ui.input("Name").props("filled autofocus") \
            .classes("w-full tg-field")
        save_name.on("keydown.enter", lambda: do_save())
        with ui.row().classes("w-full justify-end gap-2"):
            ui.button("Cancel", on_click=save_dialog.close).props("flat")
            ui.button("Save", icon="save", on_click=do_save) \
                .props("color=positive unelevated")

    # ── Prompt inspector ────────────────────────────────────────────────────
    with ui.dialog() as prompt_dialog, ui.card().classes("p-5 gap-2") \
            .style("width:1000px;max-width:96vw"):
        ui.label("Prompt for this span").classes("text-lg font-semibold")
        prompt_summary = ui.label().classes("text-xs text-muted font-mono")
        prompt_scroll = ui.scroll_area().classes("w-full").style("height:64vh")
        with ui.row().classes("w-full justify-end gap-2"):
            ui.button("Copy", icon="content_copy",
                      on_click=lambda: copy_prompt()).props("flat")
            ui.button("Close", on_click=prompt_dialog.close).props("flat")

    def copy_prompt():
        messages, _ = editor.prompt_messages()
        if not messages:
            return
        ui.clipboard.write("\n\n".join(
            f"===== {m['role'].upper()} =====\n{m['content']}" for m in messages))
        ui.notify("Prompt copied.")

    def show_prompt():
        messages, as_sent = editor.prompt_messages()
        span = editor.span
        if not messages or span is None:
            return
        chars = sum(len(m["content"]) for m in messages)
        prompt_summary.text = (
            f"{'as sent' if as_sent else 'as it would be sent now'} · "
            f"{len(messages)} messages · {chars:,} chars "
            f"(≈{est_tokens(''.join(m['content'] for m in messages)):,} tok) · "
            f"passage {len(span.original):,} chars · "
            f"context {chars - len(span.original):,} chars"
        )
        prompt_scroll.clear()
        with prompt_scroll:
            for m in messages:
                with ui.row().classes("items-center gap-2 mt-2"):
                    ui.badge(m["role"].upper()) \
                        .props(f"color={ROLE_COLOR.get(m['role'], 'grey')}") \
                        .classes("font-mono text-[10px]")
                    ui.label(f"{len(m['content']):,} chars").classes("text-xs text-muted")
                # A plain label, not markdown: the document is arbitrary text and
                # would otherwise be parsed (or break out of a code fence).
                ui.label(m["content"]).classes(
                    "text-xs font-mono whitespace-pre-wrap break-words w-full "
                    "tg-prompt-block")
        prompt_dialog.open()

    # ── Review panel ────────────────────────────────────────────────────────
    @ui.refreshable
    def review():
        job = editor.job
        if job is None:
            return
        done, total = job.progress()

        with ui.row().classes("w-full items-center gap-3 no-wrap"):
            ui.label(job.title or "Untitled").classes("text-lg font-semibold")
            ui.space()
            ui.label(f"{job.percent()}% · {done:,} of {total:,} chars") \
                .classes("text-sm text-muted font-mono")
            ui.button("Restart", icon="restart_alt", on_click=restart_job) \
                .props("flat dense color=secondary").classes("text-xs") \
                .tooltip("Reopen the new-job form with this document and "
                         "settings, to run it again with something changed. "
                         "This job is kept as it is.")
            ui.button("Export", icon="download", on_click=export) \
                .props("flat dense color=secondary").classes("text-xs")
        ui.linear_progress(value=(done / total) if total else 0.0, show_value=False) \
            .props("rounded size=8px").classes("w-full")

        b = job.budgets
        if b is not None:
            ui.label(
                f"span target {b.span_target} tok · overlap {b.overlap} tok "
                f"each side · prompt ≈ {b.span_target * 2 + b.overlap * 2} tok"
            ).classes("text-xs text-muted font-mono")

        span = editor.span
        if span is None:
            with ui.column().classes("w-full items-center justify-center gap-2 p-8"):
                ui.icon("task_alt").classes("text-5xl text-positive")
                ui.label("Every span has been decided.").classes("text-muted")
                ui.button("Export document", icon="download", on_click=export) \
                    .props("color=positive unelevated")
            return

        running = editor.running
        with ui.row().classes("w-full items-center gap-2"):
            ui.badge(f"span {editor.index + 1} of {len(job.spans)}") \
                .props("color=secondary").classes("font-mono text-[11px]") \
                .tooltip("A live count — a span whose reply hits the token cap "
                         "is split in two and retried, so the total grows. The "
                         "percentage above is measured against the document.")
            for flag in span.flags:
                ui.badge(FLAG_TEXT.get(flag, flag)).props("color=warning text-color=dark") \
                    .classes("text-[11px]")
            if span.attempts > 1:
                ui.badge(f"attempt {span.attempts}").props("outline color=grey") \
                    .classes("text-[11px]")

        with ui.row().classes("w-full gap-3 no-wrap items-stretch"):
            with ui.column().classes("flex-1 min-w-0 gap-1"):
                ui.label("ORIGINAL").classes("text-xs text-muted tracking-wide")
                ui.markdown(f"```\n{span.original}\n```") \
                    .classes("w-full text-sm tg-span-box")
            with ui.column().classes("flex-1 min-w-0 gap-1"):
                ui.label("REWRITE").classes("text-xs text-muted tracking-wide")
                page["stream_box"] = ui.markdown(f"```\n{editor.live_text()}\n```") \
                    .classes("w-full text-sm tg-span-box")

        disabled = "disable" if running else ""
        with ui.row().classes("w-full gap-2 items-center"):
            ui.button("Accept", icon="check", on_click=accept) \
                .props(f"color=positive unelevated {disabled}")
            ui.button("Edit", icon="edit", on_click=open_edit) \
                .props(f"flat color=primary {disabled}")
            ui.button("Keep original", icon="undo", on_click=keep_original) \
                .props(f"flat color=secondary {disabled}")
            ui.button("Retry", icon="refresh", on_click=retry) \
                .props(f"flat color=secondary {disabled}")
            if running:
                ui.button("Stop", icon="stop", on_click=stop) \
                    .props("color=negative unelevated").classes("text-xs") \
                    .tooltip("Halt this span. Nothing is written, and any "
                             "auto-accept run stops here.")
            # Read-only, so it stays available mid-generation.
            ui.button("Show prompt", icon="code", on_click=show_prompt) \
                .props("flat color=secondary").classes("text-xs")
            ui.space()
            ui.label("running…" if running else "").classes("text-xs text-muted")
        ui.input("Retry instruction (optional)", value=page["nudge"],
                 on_change=lambda e: page.update(nudge=e.value)) \
            .props("filled dense").classes("w-full tg-field")

    # ── Layout ──────────────────────────────────────────────────────────────
    def show_panels():
        has_job = editor.job is not None
        if has_job:
            tab["editing_job_id"] = editor.job.id
            user["editing_job_id"] = editor.job.id
        else:
            tab.pop("editing_job_id", None)
            user.pop("editing_job_id", None)
        review_card.set_visibility(has_job)
        intake_card.set_visibility(not has_job)
        if has_job:
            review.refresh()

    def new_document():
        if editor.running:
            ui.notify(en.SPAN_BUSY, type="warning")
            return
        editor.close()
        job_list.refresh()
        show_panels()

    with ui.row().classes("w-full gap-4 no-wrap").style("height: calc(100vh - 7rem)"):
        with ui.column().classes("h-full w-64 shrink-0 gap-2 no-wrap"):
            ui.button("New document", icon="add", on_click=new_document) \
                .props("color=positive unelevated").classes("w-full")
            with ui.scroll_area().classes("flex-1 w-full min-h-0 tg-list-shell"):
                job_list()

        with ui.column().classes("h-full flex-1 min-w-0 no-wrap gap-2"):
            with ui.column().classes("w-full gap-1"):
                warnings()

            intake_card = ui.card().classes("w-full flex-1 p-5 gap-3 overflow-auto")
            with intake_card:
                ui.label("New editing job").classes("text-lg font-semibold")
                title_box = _explain(
                    ui.input("Title").props("filled").classes("w-full tg-field"),
                    HELP["title"])
                # max_files caps the queue as well as the picker. Without it the
                # uploader takes file after file, listing them all, while each
                # one silently replaces the document below.
                ui.upload(on_upload=take_upload, on_rejected=reject_upload,
                          auto_upload=True, multiple=False, max_files=1,
                          label="Upload one .txt/.md file") \
                    .props("flat bordered accept=.txt,.md,.markdown") \
                    .classes("w-full tg-field")
                doc_box = _explain(
                    ui.textarea("Document")
                    .props('filled input-style="height:220px"')
                    .classes("w-full tg-field"), HELP["document"])
                with ui.row().classes("w-full gap-2 items-end no-wrap"):
                    saved_select = _explain(
                        ui.select([], label="Saved instructions",
                                  on_change=lambda e: load_saved(e.value))
                        .props("filled dense").classes("flex-1 tg-field"),
                        HELP["saved"])
                    ui.button("Save as…", icon="save", on_click=lambda: open_save()) \
                        .props("flat color=positive").classes("text-xs")
                    ui.button(icon="delete", on_click=lambda: delete_saved()) \
                        .props("flat round dense color=negative") \
                        .tooltip("Delete the selected instruction set")
                system_box = _explain(
                    ui.textarea("Editing instructions",
                                value=Instructions().system_prompt)
                    .props('filled input-style="height:120px"')
                    .classes("w-full tg-field"), HELP["system"])
                with ui.row().classes("w-full gap-3 items-end no-wrap"):
                    max_new = _explain(
                        ui.number("Max new tokens", value=700, min=64, max=4096, step=1)
                        .props("filled").classes("flex-1 tg-field"), HELP["max_new"])
                    buffer_box = _explain(
                        ui.number("Response buffer", value=150, min=0, max=2048, step=1)
                        .props("filled").classes("flex-1 tg-field"), HELP["buffer"])
                with ui.row().classes("w-full gap-3 items-end no-wrap"):
                    overlap = _explain(
                        ui.number("Chunk overlap %", value=0, min=0, max=45, step=1)
                        .props("filled").classes("flex-1 tg-field"), HELP["overlap"])
                    temperature = _explain(
                        ui.number("Temperature", value=0.2, min=0, max=2, step=0.05)
                        .props("filled").classes("flex-1 tg-field"), HELP["temperature"])
                # Every word the tool puts around the passage, laid out to be
                # edited or emptied; it competes with the system prompt above.
                with ui.expansion("Prompt wrapper — text added around your passage") \
                        .classes("w-full").props("dense"):
                    ui.label(
                        "Sent with every span. {passage}, {before}, {after}, "
                        "{first_words} and {last_words} are substituted where "
                        "they appear; leave a field empty to send nothing. Use "
                        "Show prompt during a run to see the result verbatim."
                    ).classes("text-xs text-muted mb-2")
                    instruction_box = _explain(
                        ui.textarea("Passage instruction",
                                    value=Instructions().passage_instruction)
                        .props('filled input-style="height:90px"')
                        .classes("w-full tg-field"), HELP["instruction"])
                    framing_box = _explain(
                        ui.textarea(
                            "Context framing (only used when Chunk overlap % is "
                            "above 0)", value=Instructions().context_framing)
                        .props('filled input-style="height:120px"')
                        .classes("w-full tg-field"), HELP["framing"])
                    prime_reply = _switch(
                        "Add a reply spoken as the model, before the passage",
                        HELP["prime"])
                    primed_box = _explain(
                        ui.textarea("Primed reply", value=Instructions().primed_reply)
                        .props('filled input-style="height:70px"')
                        .classes("w-full tg-field"), HELP["primed_text"])
                    primed_box.bind_visibility_from(prime_reply, "value")
                allow_deletions = _switch("Instructions may remove text",
                                          HELP["deletions"])
                auto_accept = _switch("Auto-accept spans that pass every check",
                                      HELP["auto_accept"])
                ui.button("Start editing", icon="play_arrow", on_click=create_job) \
                    .props("color=primary unelevated")

            review_card = ui.card().classes("w-full flex-1 p-5 gap-3 overflow-auto")
            with review_card:
                review()

    refresh_saved()  # also seeds a starting set on first run
    show_panels()
    watch()  # a run may already be in flight from before this page was opened
