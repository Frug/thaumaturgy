"""Model page — two columns: model/load settings (left) + generation params (right).

The left panel scans data/models for GGUFs and loads/unloads them via the engine
(spawns llama-server). The right panel shows the selected parameter set (read-only)
and, in edit mode, the parameter-set editor. VRAM is a rough pre-load estimate;
the detected max context comes from the GGUF metadata.
"""

from nicegui import app, run, ui

from thaumaturgy import appstate, engine, hf_download, store

CACHE_TYPES = ["fp16", "q8_0", "q4_0"]
CHAT_TEMPLATES = {
    "auto": "Auto",
    "gemma": "Gemma / Gemma 4",
}
REASONING_MODES = {
    "auto": "Auto",
    "off": "Off",
    "on": "On",
}

# Quantization targets offered in the download dialog (llama-quantize names).
QUANT_TYPES = ["Q3_K_M", "Q4_K_S", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]
DEFAULT_QUANT = "Q4_K_M"
ALL_QUANTS = "__all__"
UNKNOWN_QUANT = "__unknown__"

CTX_HELP = (
    "Context length. 0 = auto for llama.cpp (requires gpu-layers=-1), 8192 for "
    "other loaders. Common values: 4096, 8192, 16384, 32768, 65536, 131072."
)

VRAM_HELP = (
    "A rough estimate of GPU memory use. It needs the runtime profile to pin "
    "both values it depends on: GPU layers set to 0 or higher (not -1/auto) "
    "and a context size above 0. Otherwise it reads “auto” — llama.cpp decides "
    "those at load time, so there is nothing to estimate from."
)

# ── Generation presets (right column) ───────────────────────────────────────────
# The built-in defaults live in store.BUILTIN_PRESETS; the live, editable
# collection is persisted to <data>/presets.yaml (see store.load/save_presets).
PRESETS = store.BUILTIN_PRESETS
CUSTOM = store.CUSTOM
DEFAULT_PRESET = store.DEFAULT_PRESET
RUNTIME_TEMPLATES = store.BUILTIN_RUNTIME_TEMPLATES
DEFAULT_RUNTIME_TEMPLATE = store.DEFAULT_RUNTIME_TEMPLATE
# Slider ceiling when the model's block count can't be read.
FALLBACK_MAX_GPU_LAYERS = 100

# (key, label, min, max, step, decimals)
PARAMS = [
    ("max_new_tokens", "Max new tokens", 1, 4096, 1, 0),
    ("temperature", "Temperature", 0.0, 2.0, 0.01, 2),
    ("top_p", "Top-p", 0.0, 1.0, 0.01, 2),
    ("top_k", "Top-k", 0, 200, 1, 0),
    ("min_p", "Min-p", 0.0, 1.0, 0.01, 2),
    ("repetition_penalty", "Repetition penalty", 1.0, 1.5, 0.01, 2),
]


def _fmt(decimals: int):
    return (lambda v: f"{float(v):.{decimals}f}") if decimals else (lambda v: f"{int(v)}")


def _slider_row(label: str, minv, maxv, step, value, decimals: int, on_change=None):
    """A parameter control: name + live value on top, slider below."""
    with ui.row().classes("w-full items-center justify-between"):
        ui.label(label).classes("text-sm")
        val = ui.label().classes("text-sm font-mono text-muted")
    s = ui.slider(min=minv, max=maxv, step=step, value=value, on_change=on_change)
    val.bind_text_from(s, "value", backward=_fmt(decimals))
    return s


def _file_size_gb(name: str) -> float:
    """Size of a whole model — every shard, not just the part named."""
    try:
        return sum(p.stat().st_size for p in engine.model_files(name)) / 1e9
    except OSError:
        return 0.0


def _runtime_load_args(vals: dict) -> tuple[int, int, str, str, str, int]:
    gpu_layers = int(vals.get("gpu_layers", -1))
    ctx_size = int(vals.get("context_size", 0))
    cache_type = vals.get("cache_type", "fp16")
    if cache_type not in CACHE_TYPES:
        cache_type = "fp16"
    chat_template = vals.get("chat_template", "auto")
    if chat_template not in CHAT_TEMPLATES:
        chat_template = "auto"
    reasoning = vals.get("reasoning", "auto")
    if reasoning not in REASONING_MODES:
        reasoning = "auto"
    reasoning_budget = int(vals.get("reasoning_budget", -1))
    budget_message = str(vals.get("reasoning_budget_message",
                                  store.DEFAULT_REASONING_BUDGET_MESSAGE))
    return (gpu_layers, ctx_size, cache_type, chat_template, reasoning,
            reasoning_budget, budget_message)


def _gpu_layer_ceiling(model_name: str | None) -> int:
    """The model's offloadable layer count, or a ceiling to slide against."""
    blocks = engine.max_gpu_layers(model_name) if model_name else None
    return blocks or FALLBACK_MAX_GPU_LAYERS


def _runtime_gpu_label(vals: dict, model_name: str | None = None) -> str:
    gpu = int(vals.get("gpu_layers", -1))
    if gpu < 0:
        return "auto"
    if gpu == 0:
        return "CPU only"
    blocks = engine.max_gpu_layers(model_name) if model_name else None
    if blocks and gpu >= blocks:
        return f"{blocks} layers (all)"
    return f"{gpu} layers" + (f" of {blocks}" if blocks else "")


def _runtime_status_label(selected: str | None) -> str:
    """Load state, naming the loaded model when it isn't the selected one."""
    if not engine.server.running:
        return "Not loaded"
    loaded = engine.server.model
    if loaded and selected and loaded != selected:
        return f"Loaded ({loaded})"
    return "Loaded"


def _runtime_context_label(vals: dict) -> str:
    ctx = int(vals.get("context_size", 0))
    return "auto" if ctx == 0 else f"{ctx:,} tokens"


def _runtime_chat_template_label(vals: dict) -> str:
    return CHAT_TEMPLATES.get(vals.get("chat_template", "auto"), CHAT_TEMPLATES["auto"])


def _runtime_reasoning_label(vals: dict) -> str:
    return REASONING_MODES.get(vals.get("reasoning", "auto"), REASONING_MODES["auto"])


def _runtime_reasoning_budget_label(vals: dict) -> str:
    budget = int(vals.get("reasoning_budget", -1))
    if budget < 0:
        return "unrestricted"
    if budget == 0:
        return "immediate end"
    return f"{budget:,} tokens"


def _runtime_budget_message_label(vals: dict) -> str:
    if int(vals.get("reasoning_budget", -1)) <= 0:
        return "unused"
    message = str(vals.get("reasoning_budget_message", "")).strip()
    if not message:
        return "none (abrupt cutoff)"
    return f'"{message}"' if len(message) <= 40 else f'"{message[:37]}..."'


def _quant_filter_options(variants: list[dict]) -> dict[str, str]:
    options = {ALL_QUANTS: "All quantizations"}
    quants = sorted({v["quant"] for v in variants if v.get("quant")})
    options.update({q: q for q in quants})
    if any(not v.get("quant") for v in variants):
        options[UNKNOWN_QUANT] = "Unknown"
    return options


def _filtered_variants(variants: list[dict], quant: str | None) -> list[dict]:
    if not quant or quant == ALL_QUANTS:
        return variants
    if quant == UNKNOWN_QUANT:
        return [v for v in variants if not v.get("quant")]
    return [v for v in variants if v.get("quant") == quant]


def _model_card(bridge):
    models = engine.list_models()

    def server_output_text() -> str:
        lines = engine.server.output_lines()
        return "\n".join(lines) if lines else "No llama.cpp output yet."

    with ui.card().classes("w-full h-full p-5 gap-4 overflow-auto"):
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("Model").classes("text-lg font-semibold")
            ui.badge("llama.cpp").props("color=secondary").classes("font-mono")

        # Built early for the closures below, but filled in stages: the load
        # buttons need the download dialog, which is defined further down.
        runtime_box = ui.column().classes("tg-pset-box w-full gap-2")
        with runtime_box:
            ui.label("Runtime Settings").classes("text-xs text-muted uppercase tracking-wide")
            with ui.row().classes("w-full items-center gap-2 no-wrap"):
                if models:
                    model = ui.select(options=models, value=appstate.state.current_model
                                      if appstate.state.current_model in models else models[0]) \
                        .classes("tg-field").props("filled").style("flex:1;min-width:0")
                else:
                    model = ui.select(options=["(no models found)"], value="(no models found)") \
                        .classes("tg-field").props("filled").style("flex:1;min-width:0")
                    model.disable()
                delete_btn = ui.button(icon="delete") \
                    .props("flat dense color=negative").tooltip("Delete this model from disk")
            if not models:
                ui.label(f"Put .gguf files in {engine.models_dir()}").classes("text-xs text-muted")
                delete_btn.disable()
        bridge["model_select"] = model

        # ── Delete-model confirmation ────────────────────────────────────────
        with ui.dialog() as del_dialog, ui.card().classes("p-5 gap-3") \
                .style("width:560px;max-width:92vw"):
            ui.label("Delete model").classes("text-lg font-semibold")
            del_text = ui.label().classes("text-sm leading-relaxed")
            del_files = ui.label().classes("text-xs text-muted font-mono leading-snug")
            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=del_dialog.close).props("flat")
                del_ok = ui.button("Delete", icon="delete") \
                    .props("color=negative unelevated")

        def selected_model() -> str | None:
            v = model.value
            return v if v and not v.startswith("(") else None

        def refresh_models(value: str | None = None) -> None:
            names = engine.list_models()
            if names:
                model.set_options(names, value=value if value in names else names[0])
                model.enable()
                delete_btn.enable()
                appstate.state.current_model = model.value
                bridge["select_model_defaults"](model.value)
            else:
                model.set_options(["(no models found)"], value="(no models found)")
                model.disable()
                delete_btn.disable()
                appstate.state.current_model = None
            refresh_status()
            refresh_preview()

        def ask_delete() -> None:
            name = selected_model()
            if not name:
                return
            files = engine.model_files(name)
            if not files:
                ui.notify(f"{name} is already gone.", type="warning")
                refresh_models()
                return
            total = sum(f.stat().st_size for f in files)
            del_text.text = (
                f"Permanently delete {name}? This frees "
                f"{hf_download.human_size(total)} and cannot be undone.")
            # Spell out the whole shard set: the name alone doesn't say that
            # deleting one part takes the others with it.
            del_files.text = ("\n".join(f.name for f in files)
                              if len(files) > 1 else "")
            del_dialog.open()

        def do_delete() -> None:
            del_dialog.close()
            name = selected_model()
            if not name:
                return
            try:
                removed = engine.delete_model(name)
            except RuntimeError as exc:
                ui.notify(str(exc), type="negative")
                return
            ui.notify(f"Deleted {', '.join(removed)}", type="positive")
            refresh_models()

        delete_btn.on_click(ask_delete)
        del_ok.on_click(do_delete)
        if models:
            appstate.state.current_model = model.value
            bridge["select_model_defaults"](model.value, update_selectors=False)

        # ── Download-model dialog ────────────────────────────────────────────
        with ui.dialog() as dl_dialog, ui.card().classes("p-5 gap-3") \
                .style("width:820px;max-width:94vw"):
            ui.label("Download model").classes("text-lg font-semibold")
            ui.label("Paste a Hugging Face model URL. A repo that already has "
                     "GGUFs lists them to pick from; one with only safetensors "
                     "is converted and quantized.") \
                .classes("text-xs text-muted leading-snug")
            dl_url = ui.input(label="Hugging Face URL",
                              placeholder="https://huggingface.co/owner/model") \
                .props("filled clearable debounce=600").classes("w-full tg-field")
            dl_status = ui.label().classes("text-xs text-muted")
            with ui.row().classes("w-full items-end gap-2") as dl_filter_row:
                dl_quant_filter = ui.select(
                    options={ALL_QUANTS: "All quantizations"},
                    value=ALL_QUANTS,
                    label="Quantization",
                ).props("filled").classes("tg-field").style("min-width:220px")
            dl_filter_row.set_visibility(False)
            # Filled in by the lookup: one row per GGUF variant, or the
            # quant picker when the repo has none to choose from.
            dl_variants = ui.column().classes("w-full gap-1")
            with ui.row().classes("w-full items-end gap-2") as dl_convert_row:
                dl_quant = ui.select(options=QUANT_TYPES, value=DEFAULT_QUANT,
                                     label="Quantization") \
                    .props("filled").classes("tg-field").style("min-width:180px")
                dl_go = ui.button("Convert & Download", icon="download") \
                    .props("color=positive unelevated")
            dl_convert_row.set_visibility(False)
            with ui.row().classes("w-full justify-end gap-2"):
                dl_close = ui.button("Close", on_click=dl_dialog.close).props("flat")

        def refresh_preview():
            if "refresh_preview" in bridge:
                bridge["refresh_preview"]()

        # variant_buttons is rebuilt by render_variants; a job disables every
        # button so a second one can't start on top of the first.
        dl_state: dict = {
            "repo_id": None,
            "busy": False,
            "variant_buttons": [],
            "variants": [],
        }

        def set_busy(busy: bool) -> None:
            dl_state["busy"] = busy
            dl_url.set_enabled(not busy)
            dl_quant_filter.set_enabled(not busy)
            for btn in [dl_go, *dl_state["variant_buttons"]]:
                btn.set_enabled(not busy)

        async def run_job(fn, *args):
            """Run one download/convert job, mirroring its progress into the dialog.

            Close stays live throughout — these run for tens of minutes — so the
            busy flag is what stops a second job stacking on the first.
            """
            if dl_state["busy"]:
                return
            prog = {"msg": "Starting…"}
            timer = ui.timer(0.3, lambda: setattr(dl_status, "text", prog["msg"]))
            set_busy(True)
            try:
                name = await run.io_bound(
                    fn, *args, lambda m: prog.__setitem__("msg", m))
                ui.notify(f"Downloaded {name}", type="positive")
                model.set_options(engine.list_models(), value=name)
                model.enable()
                appstate.state.current_model = name
                bridge["select_model_defaults"](name)
                refresh_status()
                refresh_preview()
                dl_dialog.close()
            except Exception as exc:  # noqa: BLE001 - surface any failure to the user
                prog["msg"] = f"Failed: {exc}"
                ui.notify(f"Download failed: {exc}", type="negative")
            finally:
                # Render once more by hand: the timer is what was painting
                # prog, so stopping it first would strand the last message
                # (a failure, usually) behind whatever it painted before.
                timer.deactivate()
                dl_status.text = prog["msg"]
                set_busy(False)

        def render_variants() -> None:
            variants = _filtered_variants(dl_state["variants"], dl_quant_filter.value)
            dl_variants.clear()
            dl_state["variant_buttons"] = []
            with dl_variants:
                ui.label(f"{len(variants)} GGUF variant"
                         f"{'s' if len(variants) != 1 else ''} — pick one:") \
                    .classes("text-xs text-muted")
                # Repos routinely ship 20+ cuts, so cap the height and scroll.
                with ui.column().classes("w-full gap-1") \
                        .style("max-height:340px;overflow-y:auto"):
                    for v in variants:
                        with ui.row().classes("w-full items-center gap-3 p-2 rounded-lg") \
                                .style("background: rgba(52,97,140,0.08)"):
                            ui.label(v["label"]).classes("text-sm font-mono truncate") \
                                .style("flex:1;min-width:0").tooltip(v["label"])
                            if v["quant"]:
                                ui.badge(v["quant"]).props("color=primary")
                            if len(v["files"]) > 1:
                                ui.badge(f"{len(v['files'])} shards").props("color=grey-7")
                            ui.label(hf_download.human_size(v["size"])) \
                                .classes("text-xs text-muted font-mono")
                            dl_state["variant_buttons"].append(ui.button(
                                "Download", icon="download",
                                on_click=lambda _e, files=v["files"]: run_job(
                                    hf_download.fetch_variant,
                                    dl_state["repo_id"], files),
                            ).props("dense unelevated color=positive"))

        async def do_lookup():
            if dl_state["busy"]:
                return
            dl_variants.clear()
            dl_convert_row.set_visibility(False)
            dl_filter_row.set_visibility(False)
            dl_state["repo_id"] = None
            dl_state["variants"] = []
            url = (dl_url.value or "").strip()
            if not url:
                dl_status.text = ""
                return
            try:
                repo_id = hf_download.parse_repo_id(url)
            except ValueError:
                dl_status.text = "Paste a full Hugging Face model URL (owner/model)."
                return
            dl_status.text = f"Looking up {repo_id}…"
            try:
                info = await run.io_bound(hf_download.probe, url)
            except Exception as exc:  # noqa: BLE001 - surface any lookup failure
                dl_status.text = f"Lookup failed: {exc}"
                return
            dl_state["repo_id"] = info["repo_id"]
            if info["variants"]:
                dl_status.text = info["repo_id"]
                dl_state["variants"] = info["variants"]
                options = _quant_filter_options(info["variants"])
                default_quant = DEFAULT_QUANT if DEFAULT_QUANT in options else ALL_QUANTS
                dl_quant_filter.set_options(options, value=default_quant)
                dl_filter_row.set_visibility(True)
                render_variants()
            else:
                dl_status.text = (f"{info['repo_id']} ships no GGUF — it will be "
                                  "converted and quantized, which takes a while.")
                dl_convert_row.set_visibility(True)

        dl_url.on_value_change(do_lookup)
        dl_quant_filter.on_value_change(lambda _e: render_variants())
        dl_go.on_click(lambda: run_job(hf_download.convert,
                                       (dl_url.value or "").strip(), dl_quant.value))

        with runtime_box:
            with ui.row().classes("w-full gap-2"):
                load_btn = ui.button("Load model", icon="play_arrow") \
                    .props("color=positive unelevated")
                ui.button("Unload", icon="stop",
                          on_click=lambda: (engine.server.stop(), refresh_status(),
                                            refresh_preview())) \
                    .props("color=negative unelevated")
                ui.button("Download New Model", icon="download", on_click=dl_dialog.open) \
                    .props("color=primary unelevated")

            status = ui.label().classes("text-sm")
            runtime_owner = ui.label().classes("text-sm font-mono break-all")
            ui.button("Edit runtime settings", icon="edit",
                      on_click=lambda: bridge["enter_runtime_edit"]()) \
                .props("color=primary unelevated").classes("w-full")

        with ui.column().classes("tg-pset-box w-full gap-2"):
            ui.label("Parameter Set").classes("text-xs text-muted uppercase tracking-wide")
            param_set = ui.select(options=bridge["pset_options"](),
                                  value=bridge.get("start_pset", DEFAULT_PRESET)) \
                .props("filled").classes("w-full tg-field")
            with ui.row().classes("w-full gap-2 no-wrap"):
                ui.button("Use model default", icon="restart_alt",
                          on_click=lambda: bridge["use_default"]()) \
                    .props("color=primary unelevated").classes("flex-1")
                ui.button("Make default for model", icon="push_pin",
                          on_click=lambda: bridge["make_default"]()) \
                    .props("color=primary unelevated").classes("flex-1")
                ui.button(icon="edit", on_click=lambda: bridge["enter_param_edit"]()) \
                    .props("color=primary unelevated").tooltip("Edit parameter sets")
        param_set.on_value_change(
            lambda e: bridge["apply_param"](e.value) if "apply_param" in bridge else None)
        bridge["param_select"] = param_set

        def refresh_runtime_owner():
            # The dropdown above already names the model; say what's scoped to it.
            runtime_owner.text = ("Settings are saved per model."
                                  if bridge["current_model"]() else "Select a model.")

        bridge["refresh_runtime_owner"] = refresh_runtime_owner
        refresh_runtime_owner()

        with ui.column().classes("tg-pset-box w-full gap-2"):
            with ui.row().classes("w-full items-center justify-between"):
                ui.label("llama.cpp Output").classes("text-xs text-muted uppercase tracking-wide")
                ui.badge(f"last {engine.SERVER_LOG_LIMIT} lines").props("color=secondary") \
                    .classes("font-mono")
            server_output_scroll = ui.scroll_area().classes("w-full tg-server-output")
            with server_output_scroll:
                server_output = ui.label(server_output_text()) \
                    .classes("tg-server-output-text font-mono")

        shown_output = {"text": server_output.text}

        def refresh_server_output():
            if server_output.is_deleted or server_output_scroll.is_deleted:
                log_timer.cancel()
                return
            text = server_output_text()
            if text == shown_output["text"]:
                return
            shown_output["text"] = text
            server_output.text = text
            server_output_scroll.scroll_to(percent=1.0)

        # app.timer, not ui.timer: ui.timer resolves a weakref to its parent slot
        # on the way into its run loop — before it consults its own is_deleted
        # check — and raises once this client's element tree has been collected.
        # app.timer never touches the slot, so we stop it ourselves above.
        log_timer = app.timer(1.0, refresh_server_output, immediate=False)

        def refresh_status():
            if engine.server.running:
                extra = f" · ctx {engine.server.n_ctx}" if engine.server.n_ctx else ""
                status.text = f"● Loaded: {engine.server.model}{extra}"
                status.classes(replace="text-sm text-positive")
            else:
                status.text = "○ Not loaded"
                status.classes(replace="text-sm text-muted")

        async def run_load(current_model: str, gpu_layers: int, ctx_size: int,
                           cache_type: str, chat_template: str, reasoning: str,
                           reasoning_budget: int, reasoning_budget_message: str):
            load_btn.props("loading")
            ui.notify(f"Loading {current_model}…")
            try:
                await run.io_bound(engine.server.start, current_model, gpu_layers,
                                   ctx_size, cache_type, chat_template, reasoning,
                                   reasoning_budget, reasoning_budget_message)
                ui.notify(f"Loaded {current_model}")
            except Exception as exc:  # noqa: BLE001 - surface any startup failure
                ui.notify(f"Load failed: {exc}", type="negative")
            finally:
                load_btn.props(remove="loading")
                refresh_status()
                refresh_server_output()
                refresh_preview()

        async def load():
            current_model = bridge["current_model"]()
            if not current_model:
                ui.notify("No models in the models folder", type="negative")
                return
            (gpu_layers, ctx_size, cache_type, chat_template, reasoning,
             reasoning_budget, budget_message) = (
                _runtime_load_args(bridge["active_runtime"]()))
            # Backstop for hand-edited YAML; the slider can't exceed this.
            max_layers = engine.max_gpu_layers(current_model)
            if max_layers is not None:
                gpu_layers = min(gpu_layers, max_layers)
            await run_load(current_model, gpu_layers, ctx_size, cache_type,
                           chat_template, reasoning, reasoning_budget, budget_message)

        load_btn.on_click(load)

        def on_model_change(_=None):
            appstate.state.current_model = model.value
            bridge["select_model_defaults"](model.value)

        model.on_value_change(on_model_change)
        refresh_status()


def render():
    """Model page: model/profile selection, load preview, and profile editors."""
    bridge: dict = {"refresh_preview": lambda: None}
    sliders: dict[str, ui.slider] = {}

    doc = store.load_presets()
    sets: dict = doc["sets"]
    order: list = doc["order"]
    model_defaults: dict = doc["model_defaults"]

    runtime_doc = store.load_runtime_settings()
    runtime_templates: dict = runtime_doc["templates"]
    runtime_order: list = runtime_doc["order"]
    runtime_models: dict = runtime_doc["models"]

    def param_default_for_model(model_name: str | None) -> str:
        """The model's pinned parameter set, else the global default."""
        pinned = model_defaults.get(model_name)
        if pinned in sets:
            return pinned
        return DEFAULT_PRESET if DEFAULT_PRESET in sets else order[0]

    start = param_default_for_model(None)

    state = {
        "mode": "view",
        "active": start,
        "editing": start,
        # Template highlighted in the edit-mode list; templates are copied onto
        # the model, never bound to it, so this is selection only.
        "template": runtime_order[0] if runtime_order else DEFAULT_RUNTIME_TEMPLATE,
    }

    bridge["start_pset"] = start

    def persist_params():
        store.save_presets({"sets": sets, "order": order, "model_defaults": model_defaults})

    def persist_runtime():
        store.save_runtime_settings({
            "templates": runtime_templates,
            "order": runtime_order,
            "models": runtime_models,
        })

    def current_model():
        sel = bridge.get("model_select")
        v = sel.value if sel is not None else None
        return v if v and not v.startswith("(") else None

    def param_values(name: str) -> dict:
        return sets.get(name, PRESETS[DEFAULT_PRESET])

    def model_runtime(model_name: str | None) -> dict:
        """The model's own load settings, seeded from the default template."""
        if not model_name:
            return store.normalize_runtime(None)
        if model_name not in runtime_models:
            runtime_models[model_name] = store.normalize_runtime(
                runtime_templates.get(DEFAULT_RUNTIME_TEMPLATE))
        return runtime_models[model_name]

    def active_runtime():
        return model_runtime(current_model())

    def sync_active_params():
        appstate.state.current_params = dict(param_values(state["active"]))

    def refresh_selectors():
        sel = bridge.get("param_select")
        if sel is not None:
            sel.set_options(pset_options(), value=state["active"])
        if "refresh_runtime_owner" in bridge:
            bridge["refresh_runtime_owner"]()

    def refresh_preview():
        if state["mode"] == "view":
            details_panel.refresh()

    bridge["current_model"] = current_model
    bridge["active_runtime"] = active_runtime

    def select_model_defaults(model_name: str | None, update_selectors: bool = True):
        if not model_name or model_name.startswith("("):
            return
        state["active"] = param_default_for_model(model_name)
        if state["mode"] == "param_edit":
            state["editing"] = state["active"]
        bridge["start_pset"] = state["active"]
        sync_active_params()
        if update_selectors:
            refresh_selectors()
            refresh_preview()
        if state["mode"] == "runtime_edit":
            # The panel edits whichever model is selected — rebuild for the new one.
            details_panel.refresh()

    bridge["select_model_defaults"] = select_model_defaults

    def pset_options():
        d = model_defaults.get(current_model())
        return {n: (f"{n} *" if n == d else n) for n in order}

    def apply_param_view(name: str):
        if name not in sets:
            return
        state["active"] = name
        sync_active_params()
        refresh_preview()

    def make_default():
        sel = bridge.get("param_select")
        m = current_model()
        if not m:
            ui.notify("Select or load a model first", type="warning")
            return
        if sel is not None:
            model_defaults[m] = sel.value
            persist_params()
            sel.set_options(pset_options(), value=sel.value)
            ui.notify(f"'{sel.value}' is now the default for {m}")

    def use_default():
        sel = bridge.get("param_select")
        d = param_default_for_model(current_model())
        if sel is not None:
            sel.value = d

    def apply_template(name: str):
        """Copy a template's values onto the selected model."""
        m = current_model()
        if not m:
            ui.notify("Select a model first", type="warning")
            return
        if name not in runtime_templates:
            return
        state["template"] = name
        runtime_models[m] = store.normalize_runtime(runtime_templates[name])
        persist_runtime()
        details_panel.refresh()
        sets_panel.refresh()
        ui.notify(f"Applied '{name}' to {m}")

    bridge["pset_options"] = pset_options
    bridge["apply_param"] = apply_param_view
    bridge["make_default"] = make_default
    bridge["use_default"] = use_default
    bridge["refresh_preview"] = refresh_preview

    def enter_param_edit():
        state["mode"] = "param_edit"
        state["editing"] = state["active"]
        strip.classes(add="tg-edit")
        details_panel.refresh()
        sets_panel.refresh()

    def enter_runtime_edit():
        state["mode"] = "runtime_edit"
        strip.classes(add="tg-edit")
        details_panel.refresh()
        sets_panel.refresh()

    bridge["enter_param_edit"] = enter_param_edit
    bridge["enter_runtime_edit"] = enter_runtime_edit

    def exit_edit():
        if state["mode"] == "param_edit":
            if state["editing"] in sets:
                state["active"] = state["editing"]
            sync_active_params()
            persist_params()
        elif state["mode"] == "runtime_edit":
            persist_runtime()
        state["mode"] = "view"
        strip.classes(remove="tg-edit")
        refresh_selectors()
        details_panel.refresh()

    def select_set(name: str):
        state["editing"] = name
        details_panel.refresh()
        sets_panel.refresh()

    def select_template(name: str):
        state["template"] = name
        sets_panel.refresh()

    def new_set():
        i = 1
        while f"Set {i}" in sets:
            i += 1
        name = f"Set {i}"
        sets[name] = dict(PRESETS[DEFAULT_PRESET])
        order.append(name)
        persist_params()
        select_set(name)

    def save_as_template(name: str):
        """Save the selected model's current settings under a template name.

        Reusing an existing name overwrites it — that doubles as the way to
        update a template, and as the way to rename one alongside Delete.
        """
        m = current_model()
        name = (name or "").strip()
        if not m or not name:
            return
        if name not in runtime_templates:
            runtime_order.append(name)
        runtime_templates[name] = dict(model_runtime(m))
        persist_runtime()
        select_template(name)
        ui.notify(f"Saved {m}'s settings as '{name}'")

    def ask_save_template():
        if not current_model():
            ui.notify("Select a model first", type="warning")
            return
        i = 1
        while f"Template {i}" in runtime_templates:
            i += 1
        template_name.value = f"Template {i}"
        template_dialog.open()

    def rename_set(new_name: str) -> bool:
        old = state["editing"]
        new_name = (new_name or "").strip()
        if not old or not new_name or new_name == old:
            return False
        if new_name in sets:
            ui.notify(f"'{new_name}' already exists", type="negative")
            return False
        sets[new_name] = sets.pop(old)
        order[order.index(old)] = new_name
        if state["active"] == old:
            state["active"] = new_name
        for m, s in list(model_defaults.items()):
            if s == old:
                model_defaults[m] = new_name
        state["editing"] = new_name
        persist_params()
        sets_panel.refresh()
        return True


    def delete_set():
        name = state["editing"]
        if len(order) <= 1:
            ui.notify("Keep at least one parameter set", type="warning")
            return
        if name in sets:
            del sets[name]
            order.remove(name)
            for m in [m for m, s in model_defaults.items() if s == name]:
                del model_defaults[m]
            state["editing"] = order[0]
            if state["active"] == name:
                state["active"] = state["editing"]
                sync_active_params()
            persist_params()
            details_panel.refresh()
            sets_panel.refresh()

    def delete_template():
        name = state["template"]
        if len(runtime_order) <= 1:
            ui.notify("Keep at least one template", type="warning")
            return
        if name in runtime_templates:
            del runtime_templates[name]
            runtime_order.remove(name)
            state["template"] = runtime_order[0]
            persist_runtime()
            sets_panel.refresh()

    with ui.dialog() as template_dialog, ui.card().classes("p-5 gap-3") \
            .style("width:420px;max-width:92vw"):
        ui.label("Save as template").classes("text-lg font-semibold")
        template_name = ui.input(label="Template name").classes("w-full tg-field").props("filled")
        ui.label("An existing name is overwritten.").classes("text-xs text-muted")
        with ui.row().classes("w-full justify-end gap-2"):
            ui.button("Cancel", on_click=template_dialog.close).props("flat")
            ui.button("Save", icon="save",
                      on_click=lambda: (template_dialog.close(),
                                        save_as_template(template_name.value))) \
                .props("color=positive unelevated")

    with ui.dialog() as confirm_dialog, ui.card().classes("p-5 gap-3"):
        confirm_label = ui.label().classes("text-sm")
        with ui.row().classes("w-full justify-end gap-2"):
            ui.button("Cancel", on_click=confirm_dialog.close).props("flat")
            ui.button("Delete", icon="delete",
                      on_click=lambda: (confirm_dialog.close(),
                                        delete_template() if state["mode"] == "runtime_edit"
                                        else delete_set())) \
                .props("color=negative unelevated")

    def ask_delete():
        if state["mode"] == "runtime_edit" and state["template"]:
            confirm_label.text = (
                f"Delete template “{state['template']}”? This can't be undone.")
            confirm_dialog.open()
        elif state["editing"]:
            confirm_label.text = (
                f"Delete parameter set “{state['editing']}”? This can't be undone.")
            confirm_dialog.open()

    def estimate_vram(vals: dict | None = None) -> float | str:
        """Rough VRAM estimate, or a word saying why there isn't one — a
        "~ 0.0 GB" built from missing inputs reads as an answer."""
        model = current_model()
        if not model:
            return "no model"
        runtime = vals or active_runtime()
        gpu_layers = int(runtime.get("gpu_layers", -1))
        ctx_size = int(runtime.get("context_size", 0))
        if gpu_layers < 0 or ctx_size == 0:
            return "auto"
        size_gb = _file_size_gb(model)
        if not size_gb:
            return "unknown"
        frac = min(1.0, gpu_layers / _gpu_layer_ceiling(model))
        weights = size_gb * frac
        per = {"fp16": 2.0, "q8_0": 1.0, "q4_0": 0.5}[runtime.get("cache_type", "fp16")]
        kv_gb = ctx_size * 48 * per * 128 / 1e9
        return weights + kv_gb + 0.6

    def vram_label(vals: dict | None = None) -> str:
        vram = estimate_vram(vals)
        return vram if isinstance(vram, str) else f"~ {vram:.1f} GB"

    def summary_row(label: str, value: str):
        with ui.row().classes("w-full items-start justify-between gap-3 no-wrap"):
            ui.label(label).classes("text-sm text-muted")
            ui.label(value).classes("text-sm font-mono text-right break-all")

    @ui.refreshable
    def details_panel():
        with ui.card().classes("w-full h-full p-5 gap-4 overflow-auto"):
            if state["mode"] == "view":
                runtime = active_runtime()
                params = param_values(state["active"])
                with ui.row().classes("w-full items-center justify-between"):
                    ui.label("Selected Settings").classes("text-lg font-semibold")
                    ui.badge("ready").props("color=secondary").classes("font-mono")

                with ui.column().classes("w-full gap-2"):
                    ui.label("Model").classes("text-xs text-muted uppercase tracking-wide")
                    model_name = current_model()
                    summary_row("Model", model_name or "None selected")
                    summary_row("Runtime status", _runtime_status_label(model_name))
                    # From metadata, not the server: /props n_ctx is the window
                    # llama-server was launched with, not the model's limit.
                    detected = engine.trained_ctx(model_name) if model_name else None
                    summary_row("Detected max context",
                                f"{detected:,} tokens" if detected else "Unknown")
                    if engine.server.running and engine.server.n_ctx:
                        summary_row("Active context window",
                                    f"{engine.server.n_ctx:,} tokens")

                ui.separator()
                with ui.column().classes("w-full gap-2"):
                    ui.label("Runtime").classes("text-xs text-muted uppercase tracking-wide")
                    summary_row("GPU layers", _runtime_gpu_label(runtime, model_name))
                    summary_row("Context size", _runtime_context_label(runtime))
                    summary_row("KV cache type", runtime.get("cache_type", "fp16"))
                    summary_row("Chat template", _runtime_chat_template_label(runtime))
                    summary_row("Reasoning", _runtime_reasoning_label(runtime))
                    summary_row("Reasoning budget", _runtime_reasoning_budget_label(runtime))
                    summary_row("Budget message", _runtime_budget_message_label(runtime))

                ui.separator()
                with ui.column().classes("w-full gap-2"):
                    ui.label("Generation").classes("text-xs text-muted uppercase tracking-wide")
                    summary_row("Parameter set", state["active"])
                    for key, label, _minv, _maxv, _step, dec in PARAMS:
                        summary_row(label, _fmt(dec)(params.get(key, PRESETS[DEFAULT_PRESET][key])))

                ui.separator()
                with ui.row().classes("w-full items-center gap-2 p-3 rounded-lg") \
                        .style("background: rgba(52,97,140,0.10)"):
                    ui.icon("memory").classes("text-primary")
                    ui.label("Estimated VRAM").classes("text-sm")
                    with ui.icon("help_outline").classes("text-sm text-muted cursor-help"):
                        ui.tooltip(VRAM_HELP).classes("max-w-xs text-xs leading-snug")
                    ui.label(vram_label(runtime)) \
                        .classes("ml-auto text-base font-semibold font-mono")
                return

            if state["mode"] == "param_edit":
                sliders.clear()
                vals = param_values(state["editing"])
                ui.label("Editing Parameter Set").classes("text-xs text-muted uppercase tracking-wide")
                name_input = ui.input(value=state["editing"] or "") \
                    .props('filled dense input-style="font-size:1.4rem;font-weight:600"') \
                    .classes("w-full tg-field")

                def commit_rename():
                    if not rename_set(name_input.value):
                        name_input.value = state["editing"] or ""

                name_input.on("blur", commit_rename)
                name_input.on("keydown.enter", commit_rename)

                for key, label, minv, maxv, step, dec in PARAMS:
                    sliders[key] = _slider_row(label, minv, maxv, step,
                                               vals.get(key, PRESETS[DEFAULT_PRESET][key]),
                                               dec,
                                               on_change=lambda _e, k=key: on_param_change(k))

                with ui.expansion("Advanced samplers", icon="tune").classes("w-full"):
                    ui.label(
                        "typical_p, TFS, mirostat, DRY, XTC, penalties … will live here."
                    ).classes("text-xs text-muted")

                ui.button("Save", icon="save",
                          on_click=lambda: (persist_params(),
                                            ui.notify(f"Saved '{state['editing']}'",
                                                      type="positive"))) \
                    .props("color=positive unelevated").classes("mt-1")
                return

            edited_model = current_model()
            ui.label("Editing Runtime Settings").classes("text-xs text-muted uppercase tracking-wide")
            if not edited_model:
                ui.label("Select a model to edit its runtime settings.") \
                    .classes("text-sm text-muted")
                return
            vals = model_runtime(edited_model)
            ui.label(edited_model).classes("text-lg font-semibold break-all")

            controls = {}
            ceiling = _gpu_layer_ceiling(edited_model)
            controls["gpu_layers"] = _slider_row("GPU layers", -1, ceiling, 1,
                                                 min(vals.get("gpu_layers", -1), ceiling), 0)
            blocks = engine.max_gpu_layers(edited_model)
            ui.label(
                f"-1 = auto (llama.cpp fits what VRAM allows). 0 = CPU only. "
                + (f"{blocks} = every layer of this model."
                   if blocks else "The model's layer count couldn't be read.")
            ).classes("text-xs text-muted leading-snug")
            controls["context_size"] = ui.number(label="Context size",
                                                 value=vals.get("context_size", 0),
                                                 min=0, step=1024) \
                .classes("w-full tg-field").props("filled")
            ui.label(CTX_HELP).classes("text-xs text-muted leading-snug")
            controls["cache_type"] = ui.select(options=CACHE_TYPES,
                                               value=vals.get("cache_type", "fp16"),
                                               label="KV cache type") \
                .classes("w-full tg-field").props("filled")
            controls["chat_template"] = ui.select(options=CHAT_TEMPLATES,
                                                  value=vals.get("chat_template", "auto"),
                                                  label="Chat template") \
                .classes("w-full tg-field").props("filled")
            ui.label(
                "Auto uses the model's embedded/default template. Use Gemma / Gemma 4 "
                "when a model card asks for the Gemma template."
            ) \
                .classes("text-xs text-muted leading-snug")
            controls["reasoning"] = ui.select(options=REASONING_MODES,
                                              value=vals.get("reasoning", "auto"),
                                              label="Reasoning") \
                .classes("w-full tg-field").props("filled")
            ui.label(
                "Use Off for models that spend the whole reply thinking before answering."
            ).classes("text-xs text-muted leading-snug")
            controls["reasoning_budget"] = ui.number(label="Reasoning budget",
                                                     value=vals.get("reasoning_budget", -1),
                                                     min=-1, step=128) \
                .classes("w-full tg-field").props("filled")
            ui.label("-1 = unrestricted. 0 = immediate end of thinking. Above 0, "
                     "Max new tokens is added on top for the reply itself.") \
                .classes("text-xs text-muted leading-snug")
            controls["reasoning_budget_message"] = ui.input(
                label="Reasoning budget message",
                value=vals.get("reasoning_budget_message",
                               store.DEFAULT_REASONING_BUDGET_MESSAGE)) \
                .classes("w-full tg-field").props("filled")
            ui.label("The model's last thought before a spent budget cuts it off, in "
                     "its own voice, so it wraps up instead of stopping mid-sentence. "
                     "Only used when the budget is above 0.") \
                .classes("text-xs text-muted leading-snug")

            def save_runtime_edit(_=None):
                budget = controls["reasoning_budget"].value
                runtime_models[edited_model] = store.normalize_runtime({
                    "gpu_layers": controls["gpu_layers"].value,
                    "context_size": controls["context_size"].value or 0,
                    "cache_type": controls["cache_type"].value,
                    "chat_template": controls["chat_template"].value,
                    "reasoning": controls["reasoning"].value,
                    "reasoning_budget": budget if budget is not None else -1,
                    "reasoning_budget_message":
                        (controls["reasoning_budget_message"].value or "").strip(),
                })
                persist_runtime()

            for control in controls.values():
                control.on_value_change(save_runtime_edit)
            save_runtime_edit()

            ui.button("Save", icon="save",
                      on_click=lambda: (persist_runtime(),
                                        ui.notify(f"Saved settings for {edited_model}",
                                                  type="positive"))) \
                .props("color=positive unelevated").classes("mt-1")

    def on_param_change(key: str):
        if state["mode"] == "param_edit" and state["editing"] in sets and key in sliders:
            sets[state["editing"]][key] = sliders[key].value
            persist_params()

    @ui.refreshable
    def sets_list():
        with ui.list().classes("w-full"):
            if state["mode"] == "runtime_edit":
                for name in runtime_order:
                    item = ui.item(on_click=lambda n=name: apply_template(n)) \
                        .classes("tg-nav-item w-full")
                    if name == state["template"]:
                        item.classes("tg-active")
                    with item:
                        with ui.item_section().props("avatar"):
                            ui.icon("memory")
                        with ui.item_section():
                            ui.label(name).classes("font-medium")
                return
            for name in order:
                item = ui.item(on_click=lambda n=name: select_set(n)).classes("tg-nav-item w-full")
                if name == state["editing"]:
                    item.classes("tg-active")
                with item:
                    with ui.item_section().props("avatar"):
                        ui.icon("tune")
                    with ui.item_section():
                        ui.label(name).classes("font-medium")

    @ui.refreshable
    def sets_panel():
        with ui.card().classes("w-full h-full p-4 gap-2 overflow-hidden flex flex-col"):
            runtime_mode = state["mode"] == "runtime_edit"
            with ui.row().classes("w-full items-center justify-between"):
                ui.label("Templates" if runtime_mode else "Parameter sets") \
                    .classes("text-lg font-semibold")
                ui.button(icon="close", on_click=exit_edit).props("flat round dense") \
                    .tooltip("Done editing")
            if runtime_mode:
                ui.label("Click a template to copy it onto the selected model.") \
                    .classes("text-xs text-muted leading-snug")
            with ui.row().classes("w-full gap-2 no-wrap"):
                ui.button("Save as template" if runtime_mode else "New", icon="add",
                          on_click=ask_save_template if runtime_mode else new_set) \
                    .props("color=positive unelevated").classes("flex-1")
                ui.button(icon="delete", on_click=ask_delete) \
                    .props("color=negative unelevated").tooltip("Delete selected")
            with ui.scroll_area().classes("flex-1 w-full min-h-0 tg-list-shell"):
                sets_list()

    with ui.element("div").classes("w-full overflow-hidden") \
            .style("height: calc(100vh - 7rem)"):
        strip = ui.row().classes("tg-strip h-full no-wrap gap-0").style("width: 125%")
        with strip:
            with ui.element("div").classes("w-2/5 h-full px-3"):
                _model_card(bridge)
            with ui.element("div").classes("w-2/5 h-full px-3"):
                details_panel()
            with ui.element("div").classes("w-1/5 h-full px-3"):
                sets_panel()

    sync_active_params()
