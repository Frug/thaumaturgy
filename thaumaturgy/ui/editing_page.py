"""Editing page — walk a long document past the model one span at a time.

All loop and prompt logic lives in thaumaturgy.editing; this module only shows
it and collects the accept/reject decisions.
"""

import asyncio

from nicegui import ui

from thaumaturgy import appstate, editing, engine, store

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

# A span that keeps truncating is halved and retried; past this many attempts,
# stop splitting and hand it to the reviewer rather than shrinking forever.
MAX_AUTO_SPLITS = 4


def _percent(job: dict) -> int:
    done, total = editing.progress(job)
    return round(done / total * 100) if total else 0


def _rel(job: dict) -> str:
    done, total = editing.progress(job)
    return f"{_percent(job)}% · {done:,} of {total:,} chars"


def render():
    """Build the Editing page inside the current layout container."""
    page: dict = {"job": None, "index": None, "run": None, "task": None}

    def job_settings() -> dict:
        return editing.normalize_settings((page["job"] or {}).get("settings"))

    # ── Warnings ────────────────────────────────────────────────────────────
    @ui.refreshable
    def warnings():
        if not engine.server.running:
            ui.badge("No model loaded — load one on the Model page.") \
                .props("color=negative").classes("text-xs p-2 whitespace-normal")
            return
        if engine.server.thinking_enabled() and engine.server.reasoning_budget < 0:
            ui.badge(
                "This model thinks with an unrestricted budget, so a span "
                "rewrite is capped only by the context window. Set reasoning "
                "off or a budget on the Model page."
            ).props("color=warning text-color=dark") \
                .classes("text-xs p-2 whitespace-normal")
        if appstate.state.generations:
            ui.badge("A chat generation is running — editing shares the one server.") \
                .props("color=warning text-color=dark") \
                .classes("text-xs p-2 whitespace-normal")

    # ── Job list ────────────────────────────────────────────────────────────
    @ui.refreshable
    def job_list():
        jobs = store.list_jobs()
        if not jobs:
            ui.label("No documents yet — click New.").classes("text-muted text-sm p-3")
            return
        with ui.list().classes("w-full"):
            for j in jobs:
                active = page["job"] and page["job"]["id"] == j["id"]
                item = ui.item(on_click=lambda jid=j["id"]: open_job(jid)) \
                    .props("dense").classes("tg-nav-item w-full")
                if active:
                    item.classes("tg-active")
                with item, ui.item_section().classes("min-w-0"):
                    ui.label(j.get("title") or "Untitled") \
                        .classes("font-medium text-sm ellipsis w-full")
                    ui.label(f"{_percent(j)}% edited").classes("text-xs text-muted")
                with item, ui.item_section().props("side"):
                    ui.button(icon="delete",
                              on_click=lambda jid=j["id"]: remove_job(jid)) \
                        .props("flat round dense size=sm").tooltip("Delete document")

    def remove_job(job_id: str):
        if page["run"] and not page["run"]["done"]:
            ui.notify("Wait for the current span to finish.", type="warning")
            return
        store.delete_job(job_id)
        if page["job"] and page["job"]["id"] == job_id:
            page.update(job=None, index=None, run=None)
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
        # Quasar filters on `accept` before uploading; without this the file
        # just vanishes with no indication of why.
        ui.notify("That file type isn't accepted — use a .txt or .md file.",
                  type="warning")

    def create_job():
        text = doc_box.value or ""
        if not text.strip():
            ui.notify("Paste or upload a document first.", type="warning")
            return
        settings = editing.normalize_settings({
            "max_new_tokens": max_new.value,
            "temperature": temperature.value,
            "response_buffer": buffer_box.value,
            "overlap_pct": (overlap.value or 0) / 100.0,
            "auto_accept_clean": auto_accept.value,
            "allow_deletions": allow_deletions.value,
            "passage_instruction": instruction_box.value or "",
            "context_framing": framing_box.value or "",
            "primed_reply": primed_box.value or "",
            "prime_reply": prime_reply.value,
        })
        job = editing.create(title_box.value or "Untitled document", text,
                             system_box.value or editing.DEFAULT_SYSTEM_PROMPT,
                             settings)
        page.update(job=job, index=None, run=None)
        job_list.refresh()
        show_panels()
        resume()

    # ── Job open / resume ───────────────────────────────────────────────────
    def open_job(job_id: str):
        if page["run"] and not page["run"]["done"]:
            ui.notify("Wait for the current span to finish.", type="warning")
            return
        job = store.load_job(job_id)
        if job is None:
            ui.notify("That document could not be loaded.", type="negative")
            job_list.refresh()
            return
        page.update(job=editing.prepare(job), index=None, run=None)
        job_list.refresh()
        show_panels()
        resume()

    def resume():
        """Pick up at the first span that still needs a decision."""
        job = page["job"]
        if job is None:
            return
        index = editing.next_pending(job)
        if index is None:
            page["index"] = None
            review.refresh()
            return
        span = job["spans"][index]
        if span["status"] in (editing.PROPOSED, editing.FLAGGED) and span["rewritten"]:
            page["index"] = index  # already generated, just never decided
            review.refresh()
            return
        begin(index)

    def begin(index: int, nudge: str = ""):
        job = page["job"]
        if job is None:
            return
        if not engine.server.running:
            ui.notify("Load a model on the Model page first.", type="negative")
            return
        if editing.busy():
            ui.notify("The model is busy — wait for the current generation.",
                      type="warning")
            return
        # Packing sizes spans by character estimate, so a dense one can come out
        # over the reply cap; splitting now is cheaper than a wasted generation.
        if editing.oversized(job, index) and editing.split_span(job, index):
            store.save_job(job)
        page["index"] = index
        page["run"] = editing.start_span(job, index, nudge)
        # Outlives the run, so "show prompt" reports what was sent rather than
        # rebuilding it from state that may have moved on since.
        page["sent"] = (index, page["run"]["messages"])
        review.refresh()
        page["task"] = asyncio.create_task(observe(page["run"]))

    async def observe(run: dict):
        """Mirror a span run into the page until it finishes."""
        last = None
        while not run["done"]:
            await asyncio.sleep(0.15)
            box = page.get("stream_box")
            if page["run"] is not run or box is None or box.is_deleted:
                return
            if run["text"] != last:
                last = run["text"]
                box.set_content(f"```\n{run['text']}\n```")
        box = page.get("stream_box")
        if page["run"] is not run or box is None or box.is_deleted:
            return
        finish(run)

    def stop():
        """Halt the current span, and with it any auto-accept chain."""
        editing.cancel(page.get("run"))

    def finish(run: dict):
        job = page["job"]
        index = run["index"]
        if run.get("cancelled"):
            # Discard the partial reply and leave the span undecided, so
            # resuming re-runs it cleanly instead of half-editing the document.
            page["run"] = None
            ui.notify("Stopped. This span is unchanged — press Retry to run it "
                      "again, or leave the job and come back to it.")
            review.refresh()
            return
        span = editing.record_result(job, run)
        page["run"] = None
        if run.get("error"):
            ui.notify(f"Generation error: {run['error']}", type="negative")
            review.refresh()
            return
        if ("truncated" in span["flags"] and span["attempts"] <= MAX_AUTO_SPLITS
                and editing.split_span(job, index)):
            store.save_job(job)
            ui.notify("Span was too long for one reply — split and retrying.")
            begin(index)
            return
        if not span["flags"] and job_settings()["auto_accept_clean"]:
            editing.decide(job, index, editing.ACCEPTED)
            job_list.refresh()
            advance()
            return
        review.refresh()

    def advance():
        job = page["job"]
        index = editing.next_pending(job, page["index"] if page["index"] is not None else -1)
        if index is None:
            page["index"] = None
            job_list.refresh()
            review.refresh()
            return
        begin(index)

    # ── Decisions ───────────────────────────────────────────────────────────
    def accept():
        if page["index"] is None or page["run"]:
            return
        if not page["job"]["spans"][page["index"]]["rewritten"].strip():
            # Reachable after a stopped or failed run: accepting here would
            # substitute nothing for the passage and quietly delete it.
            ui.notify("There's no rewrite to accept — press Retry, or Keep "
                      "original to leave this passage alone.", type="warning")
            return
        editing.decide(page["job"], page["index"], editing.ACCEPTED)
        job_list.refresh()
        advance()

    def keep_original():
        if page["index"] is None or page["run"]:
            return
        editing.decide(page["job"], page["index"], editing.ORIGINAL)
        job_list.refresh()
        advance()

    def retry():
        if page["index"] is None or page["run"]:
            return
        begin(page["index"], page.get("nudge", ""))

    def open_edit():
        if page["index"] is None or page["run"]:
            return
        edit_area.value = page["job"]["spans"][page["index"]]["rewritten"]
        edit_dialog.open()

    def save_edit():
        text = edit_area.value or ""
        if not text.strip():
            ui.notify("The rewrite can't be empty — use Keep original instead.",
                      type="warning")
            return
        editing.decide(page["job"], page["index"], editing.ACCEPTED, text)
        edit_dialog.close()
        job_list.refresh()
        advance()

    def export():
        job = page["job"]
        if job is None:
            return
        ui.download.content(editing.assemble(job),
                            f"{job.get('title') or 'document'}-edited.txt")

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
        v = editing.normalize_instructions(vals)
        system_box.value = v["system_prompt"]
        instruction_box.value = v["passage_instruction"]
        framing_box.value = v["context_framing"]
        primed_box.value = v["primed_reply"]
        prime_reply.value = v["prime_reply"]

    def refresh_saved(select_name: str | None = None) -> None:
        sets = store.load_edit_prompts()
        if not sets:
            sets = {"Copy edit": editing.default_instructions()}
            store.save_edit_prompts(sets)
        names = sorted(sets)
        # Rebuilding the list moves `value`, which fires the change handler and
        # would overwrite whatever is in the fields. Only a real pick should load.
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

    ROLE_COLOR = {"system": "purple", "user": "primary", "assistant": "teal"}

    with ui.dialog() as prompt_dialog, ui.card().classes("p-5 gap-2") \
            .style("width:1000px;max-width:96vw"):
        ui.label("Prompt for this span").classes("text-lg font-semibold")
        prompt_summary = ui.label().classes("text-xs text-muted font-mono")
        prompt_scroll = ui.scroll_area().classes("w-full").style("height:64vh")
        with ui.row().classes("w-full justify-end gap-2"):
            ui.button("Copy", icon="content_copy",
                      on_click=lambda: copy_prompt()).props("flat")
            ui.button("Close", on_click=prompt_dialog.close).props("flat")

    def current_messages():
        """The messages for the open span — as sent, if this span has been run."""
        job, index = page["job"], page["index"]
        if job is None or index is None:
            return None, False
        sent = page.get("sent")
        if sent and sent[0] == index:
            return sent[1], True
        return editing.build_messages(job, index, page.get("nudge", "")), False

    def copy_prompt():
        messages, _ = current_messages()
        if not messages:
            return
        text = "\n\n".join(f"===== {m['role'].upper()} =====\n{m['content']}"
                           for m in messages)
        ui.clipboard.write(text)
        ui.notify("Prompt copied.")

    def show_prompt():
        messages, as_sent = current_messages()
        if not messages:
            return
        chars = sum(len(m["content"]) for m in messages)
        target = page["job"]["spans"][page["index"]]["original"]
        prompt_summary.text = (
            f"{'as sent' if as_sent else 'as it would be sent now'} · "
            f"{len(messages)} messages · {chars:,} chars "
            f"(≈{editing.est_tokens(''.join(m['content'] for m in messages)):,} tok) · "
            f"passage {len(target):,} chars · "
            f"context {chars - len(target):,} chars"
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
        job = page["job"]
        if job is None:
            return
        done, total = editing.progress(job)
        index = page["index"]

        with ui.row().classes("w-full items-center gap-3 no-wrap"):
            ui.label(job.get("title") or "Untitled").classes("text-lg font-semibold")
            ui.space()
            ui.label(_rel(job)).classes("text-sm text-muted font-mono")
            ui.button("Export", icon="download", on_click=export) \
                .props("flat dense color=secondary").classes("text-xs")
        ui.linear_progress(value=(done / total) if total else 0.0, show_value=False) \
            .props("rounded size=8px").classes("w-full")

        b = job["budgets"]
        ui.label(
            f"span target {b['span_target']} tok · overlap {b['overlap']} tok "
            f"each side · prompt ≈ {b['span_target'] * 2 + b['overlap'] * 2} tok"
        ).classes("text-xs text-muted font-mono")

        if index is None:
            with ui.column().classes("w-full items-center justify-center gap-2 p-8"):
                ui.icon("task_alt").classes("text-5xl text-positive")
                ui.label("Every span has been decided.").classes("text-muted")
                ui.button("Export document", icon="download", on_click=export) \
                    .props("color=positive unelevated")
            return

        span = job["spans"][index]
        running = page["run"] is not None
        with ui.row().classes("w-full items-center gap-2"):
            ui.badge(f"span {index + 1} of {len(job['spans'])}") \
                .props("color=secondary").classes("font-mono text-[11px]") \
                .tooltip("A live count — a span whose reply hits the token cap "
                         "is split in two and retried, so the total grows. The "
                         "percentage above is measured against the document.")
            for flag in span["flags"]:
                ui.badge(FLAG_TEXT.get(flag, flag)).props("color=warning text-color=dark") \
                    .classes("text-[11px]")
            if span["attempts"] > 1:
                ui.badge(f"attempt {span['attempts']}").props("outline color=grey") \
                    .classes("text-[11px]")

        with ui.row().classes("w-full gap-3 no-wrap items-stretch"):
            with ui.column().classes("flex-1 min-w-0 gap-1"):
                ui.label("ORIGINAL").classes("text-xs text-muted tracking-wide")
                ui.markdown(f"```\n{span['original']}\n```") \
                    .classes("w-full text-sm tg-span-box")
            with ui.column().classes("flex-1 min-w-0 gap-1"):
                ui.label("REWRITE").classes("text-xs text-muted tracking-wide")
                shown = page["run"]["text"] if running and page["run"] else span["rewritten"]
                page["stream_box"] = ui.markdown(f"```\n{shown}\n```") \
                    .classes("w-full text-sm tg-span-box")

        with ui.row().classes("w-full gap-2 items-center"):
            ui.button("Accept", icon="check", on_click=accept) \
                .props(f"color=positive unelevated {'disable' if running else ''}")
            ui.button("Edit", icon="edit", on_click=open_edit) \
                .props(f"flat color=primary {'disable' if running else ''}")
            ui.button("Keep original", icon="undo", on_click=keep_original) \
                .props(f"flat color=secondary {'disable' if running else ''}")
            ui.button("Retry", icon="refresh", on_click=retry) \
                .props(f"flat color=secondary {'disable' if running else ''}")
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
        ui.input("Retry instruction (optional)",
                 value=page.get("nudge", ""),
                 on_change=lambda e: page.update(nudge=e.value)) \
            .props("filled dense").classes("w-full tg-field")

    # ── Layout ──────────────────────────────────────────────────────────────
    def show_panels():
        has_job = page["job"] is not None
        review_card.set_visibility(has_job)
        intake_card.set_visibility(not has_job)
        if has_job:
            review.refresh()

    def new_document():
        if page["run"] and not page["run"]["done"]:
            ui.notify("Wait for the current span to finish.", type="warning")
            return
        page.update(job=None, index=None, run=None)
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
                title_box = ui.input("Title").props("filled").classes("w-full tg-field")
                ui.upload(on_upload=take_upload, on_rejected=reject_upload,
                          auto_upload=True, label="Upload a .txt/.md file") \
                    .props("flat bordered accept=.txt,.md,.markdown") \
                    .classes("w-full tg-field")
                doc_box = ui.textarea("Document") \
                    .props('filled input-style="height:220px"').classes("w-full tg-field")
                with ui.row().classes("w-full gap-2 items-end no-wrap"):
                    saved_select = ui.select(
                        [], label="Saved instructions",
                        on_change=lambda e: load_saved(e.value)) \
                        .props("filled dense").classes("flex-1 tg-field")
                    ui.button("Save as…", icon="save", on_click=lambda: open_save()) \
                        .props("flat color=positive").classes("text-xs")
                    ui.button(icon="delete", on_click=lambda: delete_saved()) \
                        .props("flat round dense color=negative") \
                        .tooltip("Delete the selected instruction set")
                system_box = ui.textarea("Editing instructions",
                                         value=editing.DEFAULT_SYSTEM_PROMPT) \
                    .props('filled input-style="height:120px"').classes("w-full tg-field")
                with ui.row().classes("w-full gap-3 items-end no-wrap"):
                    max_new = ui.number("Max new tokens", value=700, min=64, max=4096,
                                        step=1).props("filled").classes("flex-1 tg-field")
                    buffer_box = ui.number("Response buffer", value=150, min=0, max=2048,
                                           step=1).props("filled") \
                        .classes("flex-1 tg-field") \
                        .tooltip("How far under the reply cap to size each span")
                with ui.row().classes("w-full gap-3 items-end no-wrap"):
                    overlap = ui.number("Overlap %", value=0, min=0, max=45, step=1) \
                        .props("filled").classes("flex-1 tg-field") \
                        .tooltip("Surrounding text shown to the model for consistency. "
                                 "0 is the most faithful — context is the main "
                                 "cause of a model drifting out of the passage.")
                    temperature = ui.number("Temperature", value=0.2, min=0, max=2,
                                            step=0.05).props("filled") \
                        .classes("flex-1 tg-field")
                # Every word the tool puts around the passage, laid out to be
                # edited or emptied — it competes with the system prompt above,
                # so it should not be something only the code knows about.
                with ui.expansion("Prompt wrapper — text added around your passage") \
                        .classes("w-full").props("dense"):
                    ui.label(
                        "Sent with every span. {passage}, {before} and {after} "
                        "are substituted where they appear; leave a field empty "
                        "to send nothing. Use Show prompt during a run to see "
                        "the result verbatim."
                    ).classes("text-xs text-muted mb-2")
                    instruction_box = ui.textarea(
                        "Passage instruction",
                        value=editing.DEFAULT_PASSAGE_INSTRUCTION) \
                        .props('filled input-style="height:90px"') \
                        .classes("w-full tg-field")
                    framing_box = ui.textarea(
                        "Context framing (only used when Overlap % is above 0)",
                        value=editing.DEFAULT_CONTEXT_FRAMING) \
                        .props('filled input-style="height:120px"') \
                        .classes("w-full tg-field")
                    prime_reply = ui.switch(
                        "Add a reply spoken as the model, before the passage")
                    primed_box = ui.textarea(
                        "Primed reply", value=editing.DEFAULT_PRIMED_REPLY) \
                        .props('filled input-style="height:70px"') \
                        .classes("w-full tg-field")
                    primed_box.bind_visibility_from(prime_reply, "value")
                allow_deletions = ui.switch("Instructions may remove text") \
                    .tooltip("Turn on when you're asking the model to strip "
                             "content, so a shrinking span isn't treated as a "
                             "fault. Invented prose is still flagged.")
                auto_accept = ui.switch("Auto-accept spans that pass every check")
                ui.button("Start editing", icon="play_arrow", on_click=create_job) \
                    .props("color=primary unelevated")

            review_card = ui.card().classes("w-full flex-1 p-5 gap-3 overflow-auto")
            with review_card:
                review()

    refresh_saved()  # also seeds a starting set on first run
    show_panels()
