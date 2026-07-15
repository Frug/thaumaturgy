"""Model page — two columns: model/load settings (left) + generation params (right).

The left panel scans data/models for GGUFs and loads/unloads them via the engine
(spawns llama-server). The right panel shows the selected parameter set (read-only)
and, in edit mode, the parameter-set editor. VRAM is a rough pre-load estimate;
context is detected from the server after load.
"""

from nicegui import app, run, ui

from thaumaturgy import appstate, engine, hf_download, store

CACHE_TYPES = ["fp16", "q8_0", "q4_0"]

# Quantization targets offered in the download dialog (llama-quantize names).
QUANT_TYPES = ["Q3_K_M", "Q4_K_S", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]
DEFAULT_QUANT = "Q4_K_M"

CTX_HELP = (
    "Context length. 0 = auto for llama.cpp (requires gpu-layers=-1), 8192 for "
    "other loaders. Common values: 4096, 8192, 16384, 32768, 65536, 131072."
)

# ── Generation presets (right column) ───────────────────────────────────────────
# The built-in defaults live in store.BUILTIN_PRESETS; the live, editable
# collection is persisted to <data>/presets.yaml (see store.load/save_presets).
PRESETS = store.BUILTIN_PRESETS
CUSTOM = store.CUSTOM
DEFAULT_PRESET = store.DEFAULT_PRESET
RUNTIME_PROFILES = store.BUILTIN_RUNTIME_PROFILES
DEFAULT_RUNTIME_PROFILE = store.DEFAULT_RUNTIME_PROFILE

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
    try:
        return (engine.models_dir() / name).stat().st_size / 1e9
    except OSError:
        return 0.0


def _runtime_load_args(vals: dict) -> tuple[int, int, str]:
    gpu_layers = int(vals.get("gpu_layers", -1))
    ctx_size = int(vals.get("context_size", 0))
    cache_type = vals.get("cache_type", "fp16")
    if cache_type not in CACHE_TYPES:
        cache_type = "fp16"
    return gpu_layers, ctx_size, cache_type


def _runtime_gpu_label(vals: dict) -> str:
    gpu = int(vals.get("gpu_layers", -1))
    if gpu < 0:
        return "auto"
    if gpu == 0:
        return "CPU only"
    return f"{gpu} layers"


def _runtime_context_label(vals: dict) -> str:
    ctx = int(vals.get("context_size", 0))
    return "auto" if ctx == 0 else f"{ctx:,} tokens"


def _model_card(bridge):
    models = engine.list_models()

    def server_output_text() -> str:
        lines = engine.server.output_lines()
        return "\n".join(lines) if lines else "No llama.cpp output yet."

    with ui.card().classes("w-full h-full p-5 gap-4 overflow-auto"):
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("Model").classes("text-lg font-semibold")
            ui.badge("llama.cpp").props("color=secondary").classes("font-mono")

        if models:
            model = ui.select(options=models, value=appstate.state.current_model
                              if appstate.state.current_model in models else models[0]) \
                .classes("w-full tg-field").props("filled")
        else:
            model = ui.select(options=["(no models found)"], value="(no models found)") \
                .classes("w-full tg-field").props("filled")
            model.disable()
            ui.label(f"Put .gguf files in {engine.models_dir()}").classes("text-xs text-muted")
        bridge["model_select"] = model
        if models:
            appstate.state.current_model = model.value
            bridge["select_model_defaults"](model.value, update_selectors=False)

        # ── Download-model dialog ────────────────────────────────────────────
        with ui.dialog() as dl_dialog, ui.card().classes("p-5 gap-3").style("width:520px;max-width:92vw"):
            ui.label("Download model").classes("text-lg font-semibold")
            ui.label("Paste a Hugging Face model URL. If the repo only has "
                     "safetensors, it will be converted and quantized to GGUF.") \
                .classes("text-xs text-muted leading-snug")
            dl_url = ui.input(label="Hugging Face URL",
                              placeholder="https://huggingface.co/owner/model") \
                .props("filled").classes("w-full tg-field")
            dl_quant = ui.select(options=QUANT_TYPES, value=DEFAULT_QUANT,
                                 label="Quantization") \
                .props("filled").classes("w-full tg-field")
            dl_status = ui.label().classes("text-xs text-muted")
            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=dl_dialog.close).props("flat")
                dl_go = ui.button("Download New Model", icon="download") \
                    .props("color=positive unelevated")

        def refresh_preview():
            if "refresh_preview" in bridge:
                bridge["refresh_preview"]()

        async def do_download():
            url = (dl_url.value or "").strip()
            if not url:
                dl_status.text = "Enter a Hugging Face URL first."
                return
            prog = {"msg": "Starting…"}
            timer = ui.timer(0.3, lambda: setattr(dl_status, "text", prog["msg"]))
            dl_go.props("loading"); dl_go.disable()
            try:
                name = await run.io_bound(
                    hf_download.download, url, dl_quant.value,
                    lambda m: prog.__setitem__("msg", m))
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
                timer.deactivate()
                dl_go.props(remove="loading"); dl_go.enable()

        dl_go.on_click(do_download)

        with ui.row().classes("w-full gap-2 mb-4"):
            load_btn = ui.button("Load model", icon="play_arrow") \
                .props("color=positive unelevated")
            ui.button("Unload", icon="stop",
                      on_click=lambda: (engine.server.stop(), refresh_status(), refresh_preview())) \
                .props("color=negative unelevated")
            ui.button("Download New Model", icon="download", on_click=dl_dialog.open) \
                .props("color=primary unelevated")

        status = ui.label().classes("text-sm")

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

        with ui.column().classes("tg-pset-box w-full gap-2"):
            ui.label("Runtime Profile").classes("text-xs text-muted uppercase tracking-wide")
            runtime_profile = ui.select(options=bridge["runtime_options"](),
                                        value=bridge.get("start_runtime", DEFAULT_RUNTIME_PROFILE)) \
                .props("filled").classes("w-full tg-field")
            with ui.row().classes("w-full gap-2 no-wrap"):
                ui.button("Use model default", icon="restart_alt",
                          on_click=lambda: bridge["use_runtime_default"]()) \
                    .props("color=primary unelevated").classes("flex-1")
                ui.button("Make default for model", icon="push_pin",
                          on_click=lambda: bridge["make_runtime_default"]()) \
                    .props("color=primary unelevated").classes("flex-1")
                ui.button(icon="edit", on_click=lambda: bridge["enter_runtime_edit"]()) \
                    .props("color=primary unelevated").tooltip("Edit runtime profiles")
        runtime_profile.on_value_change(
            lambda e: bridge["apply_runtime"](e.value) if "apply_runtime" in bridge else None)
        bridge["runtime_select"] = runtime_profile

        with ui.column().classes("tg-pset-box w-full gap-2"):
            with ui.row().classes("w-full items-center justify-between"):
                ui.label("llama.cpp Output").classes("text-xs text-muted uppercase tracking-wide")
                ui.badge("last 500 lines").props("color=secondary").classes("font-mono")
            server_output_scroll = ui.scroll_area().classes("w-full tg-server-output")
            with server_output_scroll:
                server_output = ui.label(server_output_text()) \
                    .classes("tg-server-output-text font-mono")

        server_output_state = {"text": server_output.text}

        def refresh_server_output():
            if server_output.is_deleted or server_output_scroll.is_deleted:
                log_timer.cancel()
                return
            text = server_output_text()
            if text == server_output_state["text"]:
                return
            server_output_state["text"] = text
            server_output.text = text
            server_output_scroll.scroll_to(percent=1.0)

        log_timer = app.timer(1.0, refresh_server_output, immediate=False)
        bridge["refresh_server_output"] = refresh_server_output

        def refresh_status():
            if engine.server.running:
                extra = f" · ctx {engine.server.n_ctx}" if engine.server.n_ctx else ""
                status.text = f"● Loaded: {engine.server.model}{extra}"
                status.classes(replace="text-sm text-positive")
            else:
                status.text = "○ Not loaded"
                status.classes(replace="text-sm text-muted")

        pending_load: dict = {}
        with ui.dialog() as gpu_layers_dialog, ui.card().classes("p-5 gap-3").style("width:460px;max-width:92vw"):
            ui.label("GPU Layers Exceed Model Limit").classes("text-lg font-semibold")
            gpu_layers_warning = ui.label().classes("text-sm leading-relaxed")
            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=gpu_layers_dialog.close).props("flat")
                ok_btn = ui.button("OK").props("color=warning unelevated")

        async def run_load(current_model: str, gpu_layers: int, ctx_size: int, cache_type: str):
            load_btn.props("loading")
            ui.notify(f"Loading {current_model}…")
            try:
                await run.io_bound(engine.server.start, current_model, gpu_layers,
                                   ctx_size, cache_type)
                ui.notify(f"Loaded {current_model}")
            except Exception as exc:  # noqa: BLE001 - surface any startup failure
                ui.notify(f"Load failed: {exc}", type="negative")
            finally:
                load_btn.props(remove="loading")
                refresh_status()
                refresh_server_output()
                refresh_preview()

        async def confirm_gpu_layers():
            gpu_layers_dialog.close()
            await run_load(**pending_load)

        ok_btn.on_click(confirm_gpu_layers)

        async def load():
            current_model = bridge["current_model"]()
            if not current_model:
                ui.notify("No models in the models folder", type="negative")
                return
            gpu_layers, ctx_size, cache_type = _runtime_load_args(bridge["active_runtime"]())
            max_layers = engine.max_gpu_layers(current_model)
            if max_layers is not None and gpu_layers > max_layers:
                pending_load.clear()
                pending_load.update({
                    "current_model": current_model,
                    "gpu_layers": gpu_layers,
                    "ctx_size": ctx_size,
                    "cache_type": cache_type,
                })
                gpu_layers_warning.text = (
                    f"The selected runtime profile requests {gpu_layers} GPU layers, "
                    f"but {current_model} reports a maximum of {max_layers}. "
                    "Loading may fail or behave unexpectedly. Continue?")
                gpu_layers_dialog.open()
                return
            await run_load(current_model, gpu_layers, ctx_size, cache_type)

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
    start = DEFAULT_PRESET if DEFAULT_PRESET in sets else order[0]

    runtime_doc = store.load_runtime_profiles()
    runtime_sets: dict = runtime_doc["sets"]
    runtime_order: list = runtime_doc["order"]
    runtime_model_defaults: dict = runtime_doc["model_defaults"]
    runtime_start = (DEFAULT_RUNTIME_PROFILE if DEFAULT_RUNTIME_PROFILE in runtime_sets
                     else runtime_order[0])

    state = {
        "mode": "view",
        "active": start,
        "editing": start,
        "runtime_active": runtime_start,
        "runtime_editing": runtime_start,
    }

    bridge["start_pset"] = start
    bridge["start_runtime"] = runtime_start

    def persist_params():
        store.save_presets({"sets": sets, "order": order, "model_defaults": model_defaults})

    def persist_runtime():
        store.save_runtime_profiles({
            "sets": runtime_sets,
            "order": runtime_order,
            "model_defaults": runtime_model_defaults,
        })

    def current_model():
        sel = bridge.get("model_select")
        v = sel.value if sel is not None else None
        return v if v and not v.startswith("(") else None

    def param_values(name: str) -> dict:
        return sets.get(name, PRESETS[DEFAULT_PRESET])

    def runtime_values(name: str) -> dict:
        vals = dict(runtime_sets.get(name, RUNTIME_PROFILES[DEFAULT_RUNTIME_PROFILE]))
        vals["gpu_layers"] = int(vals.get("gpu_layers", -1))
        vals["context_size"] = int(vals.get("context_size", 0))
        if vals.get("cache_type") not in CACHE_TYPES:
            vals["cache_type"] = "fp16"
        return vals

    def active_runtime():
        return runtime_values(state["runtime_active"])

    def sync_active_params():
        appstate.state.current_params = dict(param_values(state["active"]))

    def refresh_selectors():
        sel = bridge.get("param_select")
        if sel is not None:
            sel.set_options(pset_options(), value=state["active"])
        rsel = bridge.get("runtime_select")
        if rsel is not None:
            rsel.set_options(runtime_options(), value=state["runtime_active"])

    def refresh_preview():
        if state["mode"] == "view":
            details_panel.refresh()

    bridge["current_model"] = current_model
    bridge["active_runtime"] = active_runtime

    def param_default_for_model(model_name: str | None) -> str:
        pinned = model_defaults.get(model_name)
        if pinned in sets:
            return pinned
        return DEFAULT_PRESET if DEFAULT_PRESET in sets else order[0]

    def runtime_default_for_model(model_name: str | None) -> str:
        pinned = runtime_model_defaults.get(model_name)
        if pinned in runtime_sets:
            return pinned
        return DEFAULT_RUNTIME_PROFILE if DEFAULT_RUNTIME_PROFILE in runtime_sets else runtime_order[0]

    def select_model_defaults(model_name: str | None, update_selectors: bool = True):
        if not model_name or model_name.startswith("("):
            return
        state["active"] = param_default_for_model(model_name)
        state["runtime_active"] = runtime_default_for_model(model_name)
        if state["mode"] == "param_edit":
            state["editing"] = state["active"]
        elif state["mode"] == "runtime_edit":
            state["runtime_editing"] = state["runtime_active"]
        bridge["start_pset"] = state["active"]
        bridge["start_runtime"] = state["runtime_active"]
        sync_active_params()
        if update_selectors:
            refresh_selectors()
            refresh_preview()

    bridge["select_model_defaults"] = select_model_defaults

    def pset_options():
        d = model_defaults.get(current_model())
        return {n: (f"{n} *" if n == d else n) for n in order}

    def runtime_options():
        d = runtime_model_defaults.get(current_model())
        return {n: (f"{n} *" if n == d else n) for n in runtime_order}

    def apply_param_view(name: str):
        if name not in sets:
            return
        state["active"] = name
        sync_active_params()
        refresh_preview()

    def apply_runtime_view(name: str):
        if name not in runtime_sets:
            return
        state["runtime_active"] = name
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

    def make_runtime_default():
        sel = bridge.get("runtime_select")
        m = current_model()
        if not m:
            ui.notify("Select or load a model first", type="warning")
            return
        if sel is not None:
            runtime_model_defaults[m] = sel.value
            persist_runtime()
            sel.set_options(runtime_options(), value=sel.value)
            ui.notify(f"'{sel.value}' is now the runtime default for {m}")

    def use_runtime_default():
        sel = bridge.get("runtime_select")
        d = runtime_default_for_model(current_model())
        if sel is not None:
            sel.value = d

    bridge["pset_options"] = pset_options
    bridge["runtime_options"] = runtime_options
    bridge["apply_param"] = apply_param_view
    bridge["apply_runtime"] = apply_runtime_view
    bridge["make_default"] = make_default
    bridge["use_default"] = use_default
    bridge["make_runtime_default"] = make_runtime_default
    bridge["use_runtime_default"] = use_runtime_default
    bridge["refresh_preview"] = refresh_preview

    def enter_param_edit():
        state["mode"] = "param_edit"
        state["editing"] = state["active"]
        strip.classes(add="tg-edit")
        details_panel.refresh()
        sets_panel.refresh()

    def enter_runtime_edit():
        state["mode"] = "runtime_edit"
        state["runtime_editing"] = state["runtime_active"]
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
            if state["runtime_editing"] in runtime_sets:
                state["runtime_active"] = state["runtime_editing"]
            persist_runtime()
        state["mode"] = "view"
        strip.classes(remove="tg-edit")
        refresh_selectors()
        details_panel.refresh()

    def select_set(name: str):
        state["editing"] = name
        details_panel.refresh()
        sets_panel.refresh()

    def select_runtime(name: str):
        state["runtime_editing"] = name
        details_panel.refresh()
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

    def new_runtime():
        i = 1
        while f"Profile {i}" in runtime_sets:
            i += 1
        name = f"Profile {i}"
        runtime_sets[name] = dict(RUNTIME_PROFILES[DEFAULT_RUNTIME_PROFILE])
        runtime_order.append(name)
        persist_runtime()
        select_runtime(name)

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

    def rename_runtime(new_name: str) -> bool:
        old = state["runtime_editing"]
        new_name = (new_name or "").strip()
        if not old or not new_name or new_name == old:
            return False
        if new_name in runtime_sets:
            ui.notify(f"'{new_name}' already exists", type="negative")
            return False
        runtime_sets[new_name] = runtime_sets.pop(old)
        runtime_order[runtime_order.index(old)] = new_name
        if state["runtime_active"] == old:
            state["runtime_active"] = new_name
        for m, s in list(runtime_model_defaults.items()):
            if s == old:
                runtime_model_defaults[m] = new_name
        state["runtime_editing"] = new_name
        persist_runtime()
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

    def delete_runtime():
        name = state["runtime_editing"]
        if len(runtime_order) <= 1:
            ui.notify("Keep at least one runtime profile", type="warning")
            return
        if name in runtime_sets:
            del runtime_sets[name]
            runtime_order.remove(name)
            for m in [m for m, s in runtime_model_defaults.items() if s == name]:
                del runtime_model_defaults[m]
            state["runtime_editing"] = runtime_order[0]
            if state["runtime_active"] == name:
                state["runtime_active"] = state["runtime_editing"]
            persist_runtime()
            details_panel.refresh()
            sets_panel.refresh()

    with ui.dialog() as confirm_dialog, ui.card().classes("p-5 gap-3"):
        confirm_label = ui.label().classes("text-sm")
        with ui.row().classes("w-full justify-end gap-2"):
            ui.button("Cancel", on_click=confirm_dialog.close).props("flat")
            ui.button("Delete", icon="delete",
                      on_click=lambda: (confirm_dialog.close(),
                                        delete_runtime() if state["mode"] == "runtime_edit"
                                        else delete_set())) \
                .props("color=negative unelevated")

    def ask_delete():
        if state["mode"] == "runtime_edit" and state["runtime_editing"]:
            confirm_label.text = (
                f"Delete runtime profile “{state['runtime_editing']}”? This can't be undone.")
            confirm_dialog.open()
        elif state["editing"]:
            confirm_label.text = (
                f"Delete parameter set “{state['editing']}”? This can't be undone.")
            confirm_dialog.open()

    def estimate_vram(vals: dict | None = None) -> float | None:
        model = current_model()
        if not model:
            return 0.0
        runtime = vals or active_runtime()
        gpu_layers = int(runtime.get("gpu_layers", -1))
        ctx_size = int(runtime.get("context_size", 0))
        if gpu_layers < 0 or ctx_size == 0:
            return None
        frac = min(1.0, gpu_layers / 100)
        weights = _file_size_gb(model) * frac
        per = {"fp16": 2.0, "q8_0": 1.0, "q4_0": 0.5}[runtime.get("cache_type", "fp16")]
        kv_gb = ctx_size * 48 * per * 128 / 1e9
        return weights + kv_gb + 0.6

    def vram_label(vals: dict | None = None) -> str:
        vram = estimate_vram(vals)
        return "auto" if vram is None else f"~ {vram:.1f} GB"

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
                    summary_row("Model", current_model() or "None selected")
                    summary_row("Runtime status", "Loaded" if engine.server.running else "Not loaded")
                    detected = engine.server.n_ctx or (engine.trained_ctx(current_model())
                                                       if current_model() else None)
                    summary_row("Detected max context",
                                f"{detected:,} tokens" if detected else "Unknown")

                ui.separator()
                with ui.column().classes("w-full gap-2"):
                    ui.label("Runtime").classes("text-xs text-muted uppercase tracking-wide")
                    summary_row("Profile", state["runtime_active"])
                    summary_row("GPU layers", _runtime_gpu_label(runtime))
                    summary_row("Context size", _runtime_context_label(runtime))
                    summary_row("KV cache type", runtime.get("cache_type", "fp16"))
                    summary_row("Estimated VRAM", vram_label(runtime))

                ui.separator()
                with ui.column().classes("w-full gap-2"):
                    ui.label("Generation").classes("text-xs text-muted uppercase tracking-wide")
                    summary_row("Parameter set", state["active"])
                    for key, label, _minv, _maxv, _step, dec in PARAMS:
                        summary_row(label, _fmt(dec)(params.get(key, PRESETS[DEFAULT_PRESET][key])))
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

            vals = runtime_values(state["runtime_editing"])
            ui.label("Editing Runtime Profile").classes("text-xs text-muted uppercase tracking-wide")
            name_input = ui.input(value=state["runtime_editing"] or "") \
                .props('filled dense input-style="font-size:1.4rem;font-weight:600"') \
                .classes("w-full tg-field")

            def commit_runtime_rename():
                if not rename_runtime(name_input.value):
                    name_input.value = state["runtime_editing"] or ""

            name_input.on("blur", commit_runtime_rename)
            name_input.on("keydown.enter", commit_runtime_rename)

            controls = {}
            controls["gpu_layers"] = _slider_row("GPU layers", -1, 100, 1,
                                                 vals.get("gpu_layers", -1), 0)
            ui.label("-1 = auto. 0 = CPU only. Higher values offload that many layers.") \
                .classes("text-xs text-muted")
            controls["context_size"] = ui.number(label="Context size",
                                                 value=vals.get("context_size", 0),
                                                 min=0, step=1024) \
                .classes("w-full tg-field").props("filled")
            ui.label(CTX_HELP).classes("text-xs text-muted leading-snug")
            controls["cache_type"] = ui.select(options=CACHE_TYPES,
                                               value=vals.get("cache_type", "fp16"),
                                               label="KV cache type") \
                .classes("w-full tg-field").props("filled")

            with ui.row().classes("w-full items-center gap-2 mt-1 p-3 rounded-lg") \
                    .style("background: rgba(52,97,140,0.10)"):
                ui.icon("memory").classes("text-primary")
                ui.label("Estimated VRAM").classes("text-sm")
                vram = ui.label().classes("ml-auto text-base font-semibold font-mono")

            def save_runtime_edit(_=None):
                runtime_sets[state["runtime_editing"]] = {
                    "gpu_layers": int(controls["gpu_layers"].value),
                    "context_size": int(controls["context_size"].value or 0),
                    "cache_type": controls["cache_type"].value,
                }
                persist_runtime()
                vram.text = vram_label(runtime_sets[state["runtime_editing"]])

            for control in controls.values():
                control.on_value_change(save_runtime_edit)
            save_runtime_edit()

            ui.button("Save", icon="save",
                      on_click=lambda: (persist_runtime(),
                                        ui.notify(f"Saved '{state['runtime_editing']}'",
                                                  type="positive"))) \
                .props("color=positive unelevated").classes("mt-1")

    def on_param_change(key: str):
        if state["mode"] == "param_edit" and state["editing"] in sets and key in sliders:
            sets[state["editing"]][key] = sliders[key].value
            persist_params()

    @ui.refreshable
    def sets_list():
        with ui.list().classes("w-full gap-1"):
            if state["mode"] == "runtime_edit":
                for name in runtime_order:
                    item = ui.item(on_click=lambda n=name: select_runtime(n)) \
                        .classes("tg-nav-item w-full")
                    if name == state["runtime_editing"]:
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
                ui.label("Runtime profiles" if runtime_mode else "Parameter sets") \
                    .classes("text-lg font-semibold")
                ui.button(icon="close", on_click=exit_edit).props("flat round dense") \
                    .tooltip("Done editing")
            with ui.row().classes("w-full gap-2 no-wrap"):
                ui.button("New", icon="add", on_click=new_runtime if runtime_mode else new_set) \
                    .props("color=positive unelevated").classes("flex-1")
                ui.button(icon="delete", on_click=ask_delete) \
                    .props("color=negative unelevated").tooltip("Delete selected")
            with ui.scroll_area().classes("flex-1 w-full rounded-lg p-1") \
                    .style("background: rgba(52,97,140,0.08)"):
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
