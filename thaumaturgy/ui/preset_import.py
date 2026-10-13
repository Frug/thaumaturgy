"""Dialog importing a SillyTavern preset, from a link or an uploaded file."""

from collections.abc import Callable

from nicegui import run, ui

from thaumaturgy import sillytavern


def build(add_set: Callable[[str, dict], str]) -> Callable[[], None]:
    """Build the dialog in the current page and return a function opening it.

    ``add_set(name, values)`` saves the imported parameter set and returns the
    name it was saved under.
    """

    def finish(imported: sillytavern.ImportedPreset):
        if not dialog.value:
            return  # Cancelled while the fetch ran.
        dialog.close()
        name = add_set(imported.name, imported.values)
        msg = f"Imported '{name}'."
        if imported.clamped:
            msg += " Clamped to the slider range: " + ", ".join(imported.clamped) + "."
        if imported.skipped:
            msg += " Not carried over: " + ", ".join(imported.skipped) + "."
        ui.notify(msg, type="positive", multi_line=True, timeout=10000)

    async def from_url():
        url = (url_box.value or "").strip()
        # Enter in the field bypasses the disabled button.
        if not url or not fetch_button.enabled:
            return
        fetch_button.disable()
        try:
            imported = await run.io_bound(sillytavern.fetch, url)
        except Exception as exc:  # noqa: BLE001
            ui.notify(f"Could not import: {exc}", type="negative")
            return
        finally:
            fetch_button.enable()
        finish(imported)

    async def from_upload(e):
        # Broad catch: NiceGUI reports an upload handler's exception as success.
        try:
            imported = sillytavern.parse_text(await e.file.read(),
                                              sillytavern.stem_name(e.file.name))
        except Exception as exc:  # noqa: BLE001
            ui.notify(f"Could not import {e.file.name}: {exc}", type="negative")
            # max-files counts the failed file, which would block a retry.
            upload_box.reset()
            return
        finish(imported)

    def open_dialog():
        url_box.value = ""
        upload_box.reset()
        dialog.open()

    with ui.dialog() as dialog, ui.card().classes("p-5 gap-3") \
            .style("width:480px;max-width:92vw"):
        ui.label("Import SillyTavern preset").classes("text-lg font-semibold")
        ui.label("Temperature, top-p, top-k, min-p, repetition penalty and "
                 "response length become a new parameter set. Other samplers, "
                 "templates and prompts are not imported.") \
            .classes("text-xs text-muted leading-snug")
        with ui.row().classes("w-full gap-2 items-center no-wrap"):
            url_box = ui.input(label="Link to a .json preset") \
                .props("filled dense").classes("flex-1 tg-field") \
                .on("keydown.enter", from_url)
            fetch_button = ui.button("Fetch", icon="download", on_click=from_url) \
                .props("color=positive unelevated")
        upload_box = ui.upload(
            on_upload=from_upload, auto_upload=True, multiple=False, max_files=1,
            max_file_size=sillytavern.MAX_BYTES, label="Or upload a .json file",
            on_rejected=lambda _: ui.notify(
                "Not accepted: one .json file under 2 MB at a time.", type="warning")) \
            .props("flat bordered accept=.json").classes("w-full tg-field")
        with ui.row().classes("w-full justify-end"):
            ui.button("Cancel", on_click=dialog.close).props("flat")

    return open_dialog
