"""Settings page: app preferences that persist to <data>/app_config.yaml."""

import os
from pathlib import Path

from nicegui import ui

from thaumaturgy import paths, store
from thaumaturgy.lang import en


def render() -> None:
    """Build the Settings page inside the current layout container."""
    env_override = (os.environ.get("THAUM_LOG_DIR") or "").strip()

    with ui.card().classes("w-full max-w-3xl mx-auto p-8 gap-5"):
        ui.label("Settings").classes("text-2xl font-semibold")

        with ui.column().classes("tg-pset-box w-full gap-2"):
            ui.label("Chat compaction").classes(
                "text-xs text-muted uppercase tracking-wide")

            def set_divider(value: bool) -> None:
                store.save_compaction_divider(value)
                ui.notify("Compaction divider shown" if value
                          else "Compaction divider hidden", type="positive")

            ui.switch("Show the compaction divider in chats",
                      value=store.compaction_divider(),
                      on_change=lambda e: set_divider(e.value)) \
                .classes("text-sm")
            ui.label(en.COMPACTION_HELP).classes("text-xs text-muted leading-snug")

        with ui.column().classes("tg-pset-box w-full gap-2"):
            ui.label("Diagnostic logs").classes(
                "text-xs text-muted uppercase tracking-wide")
            log_input = ui.input(
                label="Log directory",
                value=env_override or store.log_dir_setting(),
                placeholder=str(Path.home() / "thaumaturgy-logs"),
            ).classes("w-full tg-field").props("filled clearable")
            ui.label(en.LOG_HELP).classes("text-xs text-muted leading-snug")

            status = ui.label().classes("text-sm")

            def refresh_status() -> None:
                current = paths.log_dir()
                if current is None:
                    status.text = "○ Logging off"
                    status.classes(replace="text-sm text-muted")
                else:
                    status.text = f"● Writing to {current}"
                    status.classes(replace="text-sm text-positive break-all")

            def save() -> None:
                raw = (log_input.value or "").strip()
                if raw:
                    path = Path(raw).expanduser()
                    try:
                        path.mkdir(parents=True, exist_ok=True)
                    except OSError as exc:
                        ui.notify(f"Can't use that directory: {exc}", type="negative")
                        return
                store.save_log_dir(raw)
                refresh_status()
                ui.notify(f"Logging to {raw}" if raw else "Logging off",
                          type="positive")

            def turn_off() -> None:
                log_input.value = ""
                save()

            with ui.row().classes("w-full gap-2"):
                save_btn = ui.button("Save", icon="save", on_click=save) \
                    .props("color=positive unelevated")
                off_btn = ui.button("Turn off", icon="block", on_click=turn_off) \
                    .props("color=negative unelevated")

            if env_override:
                log_input.disable()
                save_btn.disable()
                off_btn.disable()
                ui.label("Set by $THAUM_LOG_DIR, which overrides the saved "
                         "setting. Unset it to manage the directory here.") \
                    .classes("text-xs text-muted leading-snug")

            refresh_status()
