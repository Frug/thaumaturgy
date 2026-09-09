"""Settings page: app preferences that persist to <data>/app_config.yaml."""

import os
import secrets
from pathlib import Path

from nicegui import ui

from thaumaturgy import paths, store
from thaumaturgy.lang import en

# The editable parts of compaction.yaml, in the order the summarizer sees them.
RECAP_FIELDS = (
    ("system", "Summarizer role (system message)", 120),
    ("instruction", "Recap instruction", 320),
    ("carry", "Heading for the previous recap", 90),
)


def render() -> None:
    """Build the Settings page inside the current layout container."""
    env_override = (os.environ.get("THAUM_LOG_DIR") or "").strip()

    with ui.card().classes("w-full max-w-3xl mx-auto p-8 gap-5"):
        ui.label("Settings").classes("text-2xl font-semibold")

        with ui.column().classes("tg-pset-box w-full gap-2"):
            ui.label("Model storage").classes(
                "text-xs text-muted uppercase tracking-wide")
            default_models_dir = paths.sub_dir("models")
            models_input = ui.input(
                label="Models directory",
                value=store.models_dir_setting(),
                placeholder=str(default_models_dir),
            ).classes("w-full tg-field").props("filled clearable")
            ui.label(
                "The Model page finds, downloads, loads, and deletes GGUF files "
                "here. Leave blank to use the models folder in the data "
                "directory. A model already loaded keeps running until you "
                "unload it."
            ).classes("text-xs text-muted leading-snug")
            models_status = ui.label().classes("text-sm text-muted break-all")

            def refresh_models_status() -> None:
                raw = store.models_dir_setting()
                current = Path(raw).expanduser() if raw else default_models_dir
                models_status.text = f"Using {current}"

            def save_models() -> None:
                raw = (models_input.value or "").strip()
                target = Path(raw).expanduser() if raw else default_models_dir
                try:
                    target.mkdir(parents=True, exist_ok=True)
                except OSError as exc:
                    ui.notify(f"Can't use that directory: {exc}", type="negative")
                    return
                store.save_models_dir(raw)
                refresh_models_status()
                ui.notify(f"Models directory set to {target}", type="positive")

            def use_default_models() -> None:
                models_input.value = ""
                save_models()

            with ui.row().classes("w-full gap-2"):
                ui.button("Save", icon="save", on_click=save_models) \
                    .props("color=positive unelevated")
                ui.button("Use default", icon="restart_alt",
                          on_click=use_default_models).props("flat")

            refresh_models_status()

        with ui.column().classes("tg-pset-box w-full gap-2"):
            ui.label("Network and OpenAI-compatible API").classes(
                "text-xs text-muted uppercase tracking-wide")
            network = store.network_settings()
            app_host_env = (os.environ.get("THAUM_HOST") or "").strip()
            app_port_env = (os.environ.get("THAUM_PORT") or "").strip()
            llama_host_env = (os.environ.get("THAUM_LLAMA_HOST") or "").strip()
            llama_port_env = (os.environ.get("THAUM_LLAMA_PORT") or "").strip()
            llama_key_env = (os.environ.get("THAUM_LLAMA_API_KEY") or "").strip()

            def host_allows_network(host: str) -> bool:
                return host not in {"127.0.0.1", "::1", "localhost"}

            ui.label("Web UI").classes("text-sm font-semibold mt-1")
            app_network_switch = ui.switch(
                "Allow network access to the web UI",
                value=(host_allows_network(app_host_env) if app_host_env
                       else network["app_network_access"]),
            ).classes("text-sm")
            app_port_input = ui.input(
                "Web UI port",
                value=app_port_env or str(network["app_port"]),
            ).classes("w-full tg-field").props("filled")
            ui.label(
                "The web UI has no authentication. Enabling network access "
                "lets other devices that can reach this computer manage "
                "models and settings."
            ).classes("text-xs text-warning leading-snug")

            ui.label("Model API").classes("text-sm font-semibold mt-3")
            llama_network_switch = ui.switch(
                "Allow network access to the model API",
                value=(host_allows_network(llama_host_env) if llama_host_env
                       else network["llama_network_access"]),
            ).classes("text-sm")
            llama_port_input = ui.input(
                "Model API port",
                value=llama_port_env or (
                    str(network["llama_port"]) if network["llama_port"] else ""),
                placeholder="Random free port",
            ).classes("w-full tg-field").props("filled clearable")
            llama_key_input = ui.input(
                "Model API key",
                value=llama_key_env or network["llama_api_key"],
                password=True,
                password_toggle_button=True,
            ).classes("w-full tg-field").props("filled clearable autocomplete=off")

            def generate_key() -> None:
                llama_key_input.value = secrets.token_hex(32)

            ui.button("Generate new key", icon="key", on_click=generate_key).props("flat")
            ui.label(
                "Network access listens on all IPv4 interfaces (0.0.0.0); off "
                "listens only on this computer (127.0.0.1). The model API "
                "requires a key when network access is enabled. Web UI changes "
                "apply after an app restart; model API changes apply the next "
                "time a model is loaded."
            ).classes("text-xs text-muted leading-snug")

            network_status = ui.label().classes("text-sm text-muted")

            def read_port(raw: str, label: str, optional: bool = False) -> int | None:
                raw = raw.strip()
                if optional and not raw:
                    return None
                try:
                    port = int(raw)
                except ValueError:
                    ui.notify(f"{label} must be an integer", type="negative")
                    return None
                if not 1 <= port <= 65535:
                    ui.notify(f"{label} must be between 1 and 65535",
                              type="negative")
                    return None
                return port

            def save_network() -> None:
                raw_app_port = (app_port_input.value or "").strip()
                app_port = read_port(raw_app_port, "Web UI port")
                if app_port is None:
                    return

                raw_port = (llama_port_input.value or "").strip()
                llama_port = read_port(raw_port, "Model API port", optional=True)
                if raw_port and llama_port is None:
                    return

                api_key = (llama_key_input.value or "").strip()
                if llama_network_switch.value and not api_key:
                    ui.notify("Network access to the model API requires an API key",
                              type="negative")
                    return

                store.save_network_settings(
                    app_network_switch.value, app_port,
                    llama_network_switch.value, llama_port, api_key)
                network_status.text = (
                    "Saved. Restart the app for the admin UI address; reload "
                    "the model for model API changes.")
                ui.notify("Network settings saved", type="positive")

            ui.button("Save network settings", icon="save", on_click=save_network) \
                .props("color=positive unelevated")

            overridden = [name for name, value in (
                ("web UI network access", app_host_env),
                ("web UI port", app_port_env),
                ("model API network access", llama_host_env),
                ("model API port", llama_port_env),
                ("model API key", llama_key_env),
            ) if value]
            if overridden:
                ui.label(
                    "Environment variables currently override: "
                    + ", ".join(overridden)
                    + ". Saved values take effect after those variables are unset."
                ).classes("text-xs text-warning leading-snug")

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

            def set_strategy(value: str) -> None:
                store.save_compaction_strategy(value)
                ui.notify("Recaps written in " + ("several passes"
                          if value == "passes" else "one pass"), type="positive")

            ui.label("Recap detail").classes(
                "text-xs text-muted uppercase tracking-wide mt-3")
            ui.radio({"single": "One pass — fastest",
                      "passes": "Several passes — more detail, slower"},
                     value=store.compaction_strategy(),
                     on_change=lambda e: set_strategy(e.value)) \
                .props("dense").classes("text-sm")
            ui.label(en.COMPACTION_STRATEGY_HELP).classes(
                "text-xs text-muted leading-snug")

        with ui.column().classes("tg-pset-box w-full gap-2"):
            ui.label("Recap instructions").classes(
                "text-xs text-muted uppercase tracking-wide")
            ui.label(en.RECAP_PROMPT_HELP).classes("text-xs text-muted leading-snug")

            doc = store.load_compaction_prompt()
            boxes = {
                key: ui.textarea(label, value=doc[key]).classes("w-full tg-field")
                        .props(f'filled input-style="height:{height}px"')
                for key, label, height in RECAP_FIELDS
            }
            ui.label(en.RECAP_PROMPT_PLACEHOLDERS).classes(
                "text-xs text-muted leading-snug")

            def save_prompt() -> None:
                edited = {key: (box.value or "").strip() for key, box in boxes.items()}
                # A blank field would be filled from the default on the next
                # load, so refuse rather than silently discard the edit.
                if not all(edited.values()):
                    ui.notify(en.RECAP_PROMPT_EMPTY, type="negative")
                    return
                store.save_compaction_prompt({**store.load_compaction_prompt(), **edited})
                ui.notify("Recap instructions saved", type="positive")

            def load_defaults() -> None:
                defaults = store.default_compaction_prompt()
                for key, box in boxes.items():
                    box.value = defaults[key]
                ui.notify(en.RECAP_PROMPT_RESTORED, type="info")

            with ui.row().classes("w-full gap-2"):
                ui.button("Save instructions", icon="save", on_click=save_prompt) \
                    .props("color=positive unelevated")
                ui.button("Restore defaults", icon="restart_alt",
                          on_click=load_defaults).props("flat")

        with ui.column().classes("tg-pset-box w-full gap-2"):
            ui.label("Diagnostic logs").classes(
                "text-xs text-muted uppercase tracking-wide")
            log_input = ui.input(
                label="Log directory",
                value=env_override or store.log_dir_setting(),
                placeholder=str(Path.home() / "thaumaturgy-logs"),
            ).classes("w-full tg-field").props("filled clearable")
            ui.label(en.LOG_HELP).classes("text-xs text-muted leading-snug")

            def set_stream_stats(value: bool) -> None:
                store.save_verbose_stream_log(value)
                ui.notify("Logging every chat request" if value
                          else "Logging blank replies only", type="positive")

            ui.switch("Log a summary of every chat request",
                      value=store.verbose_stream_log(),
                      on_change=lambda e: set_stream_stats(e.value)) \
                .classes("text-sm")
            ui.label(en.STREAM_STATS_HELP).classes("text-xs text-muted leading-snug")

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
