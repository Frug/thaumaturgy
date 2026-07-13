"""Model page — two columns: model/load settings (left) + generation params (right).

The left panel scans data/models for GGUFs and loads/unloads them via the engine
(spawns llama-server). The right panel shows the selected parameter set (read-only)
and, in edit mode, the parameter-set editor. VRAM is a rough pre-load estimate;
context is detected from the server after load.
"""

from nicegui import run, ui

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


def _model_card(bridge):
    models = engine.list_models()
    with ui.card().classes("w-full h-full p-5 gap-4 overflow-auto"):
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("Model").classes("text-lg font-semibold")
            ui.badge("llama.cpp").props("color=secondary").classes("font-mono")

        if models:
            model = ui.select(options=models, value=appstate.state.current_model
                              if appstate.state.current_model in models else models[0],
                              label="Downloaded model").classes("w-full tg-field").props("filled")
        else:
            model = ui.select(options=["(no models found)"], value="(no models found)",
                              label="Downloaded model").classes("w-full tg-field").props("filled")
            model.disable()
            ui.label(f"Put .gguf files in {engine.models_dir()}").classes("text-xs text-muted")
        bridge["model_select"] = model  # lets per-model preset defaults resolve

        status = ui.label().classes("text-sm")

        with ui.column().classes("tg-pset-box w-full gap-2"):
            param_set = ui.select(options=bridge["pset_options"](),
                                  value=bridge.get("start_pset", DEFAULT_PRESET),
                                  label="Parameter set") \
                .props("filled").classes("w-full tg-field")
            with ui.row().classes("w-full gap-2 no-wrap"):
                ui.button("Use model default", icon="restart_alt",
                          on_click=lambda: bridge["use_default"]()) \
                    .props("color=primary unelevated").classes("flex-1")
                ui.button("Make default for model", icon="push_pin",
                          on_click=lambda: bridge["make_default"]()) \
                    .props("color=primary unelevated").classes("flex-1")
        param_set.on_value_change(
            lambda e: bridge["apply"](e.value) if "apply" in bridge else None)
        bridge["param_select"] = param_set  # so edit-mode can sync the selection back

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
                dl_go = ui.button("Download", icon="download") \
                    .props("color=positive unelevated")

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
                refresh_status()
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
                      on_click=lambda: (engine.server.stop(), refresh_status())) \
                .props("color=negative unelevated")
            ui.button("Download", icon="download", on_click=dl_dialog.open) \
                .props("color=primary unelevated")

        with ui.row().classes("w-full items-center justify-between"):
            ui.label("GPU layers").classes("text-sm")
            gpu_val = ui.label().classes("text-sm font-mono text-muted")
        gpu = ui.slider(min=0, max=100, step=1, value=100)
        ui.label("100 = all layers on GPU.").classes("text-xs text-muted")

        ctx = ui.number(label="Context size", value=8192, min=0, step=1024) \
            .classes("w-full tg-field").props("filled")
        ctx_detected = ui.label().classes("text-xs text-muted")
        ui.label(CTX_HELP).classes("text-xs text-muted leading-snug")

        cache = ui.select(options=CACHE_TYPES, value="fp16", label="KV cache type") \
            .classes("w-full tg-field").props("filled")

        with ui.row().classes("w-full items-center gap-2 mt-1 p-3 rounded-lg") \
                .style("background: rgba(52,97,140,0.10)"):
            ui.icon("memory").classes("text-primary")
            ui.label("Estimated VRAM").classes("text-sm")
            vram = ui.label().classes("ml-auto text-base font-semibold font-mono")

        def estimate_vram() -> float:
            if not models:
                return 0.0
            frac = min(1.0, int(gpu.value) / 100)
            weights = _file_size_gb(model.value) * frac
            eff_ctx = int(ctx.value or 0) or (engine.server.n_ctx or 8192)
            per = {"fp16": 2.0, "q8_0": 1.0, "q4_0": 0.5}[cache.value]
            kv_gb = eff_ctx * 48 * per * 128 / 1e9  # rough KV-cache term
            return weights + kv_gb + 0.6

        def refresh():
            gpu_val.text = f"{int(gpu.value)}" + ("  (all)" if int(gpu.value) >= 100 else "")
            if engine.server.running and engine.server.n_ctx:
                detected = engine.server.n_ctx
            else:
                detected = engine.trained_ctx(model.value) if models else None
            ctx_detected.text = (f"Detected max for model: {detected:,} tokens"
                                 if detected else "Detected max for model: — (unknown)")
            vram.text = f"~ {estimate_vram():.1f} GB"

        def refresh_status():
            if engine.server.running:
                extra = f" · ctx {engine.server.n_ctx}" if engine.server.n_ctx else ""
                status.text = f"● Loaded: {engine.server.model}{extra}"
                status.classes(replace="text-sm text-positive")
            else:
                status.text = "○ Not loaded"
                status.classes(replace="text-sm text-muted")
            refresh()

        async def load():
            if not models:
                ui.notify("No models in the models folder", type="negative")
                return
            name = model.value
            gpu_layers = 999 if int(gpu.value) >= 100 else int(gpu.value)
            load_btn.props("loading")
            ui.notify(f"Loading {name}…")
            try:
                await run.io_bound(engine.server.start, name, gpu_layers,
                                   int(ctx.value or 8192), cache.value)
                ui.notify(f"Loaded {name}")
            except Exception as exc:  # noqa: BLE001 - surface any startup failure
                ui.notify(f"Load failed: {exc}", type="negative")
            finally:
                load_btn.props(remove="loading")
                refresh_status()

        load_btn.on_click(load)
        if models:
            appstate.state.current_model = model.value

        def on_model_change(_=None):
            appstate.state.current_model = model.value
            sel = bridge.get("param_select")
            if sel is not None:
                sel.set_options(bridge["pset_options"](), value=sel.value)
            refresh()
        model.on_value_change(on_model_change)
        gpu.on_value_change(lambda _: refresh())
        ctx.on_value_change(lambda _: refresh())
        cache.on_value_change(lambda _: refresh())
        refresh_status()


def render():
    """Model page. Two modes on a sliding 3-panel filmstrip:

    view:  [ Model ][ Parameters(read-only) ]        (SetsList off-screen right)
    edit:  [ Parameters(editable) ][ SetsList ]      (Model off-screen left)
    """
    bridge: dict = {}
    sliders: dict[str, ui.slider] = {}

    # Parameter sets loaded from (and saved back to) <data>/presets.yaml.
    # model_defaults maps a model filename -> the set that's its default.
    doc = store.load_presets()
    sets: dict = doc["sets"]
    order: list = doc["order"]
    model_defaults: dict = doc["model_defaults"]
    start = DEFAULT_PRESET if DEFAULT_PRESET in sets else order[0]
    state = {"mode": "view", "active": start, "editing": start, "loading": False}

    bridge["start_pset"] = start

    def persist():
        store.save_presets({"sets": sets, "order": order, "model_defaults": model_defaults})

    def current_model():
        sel = bridge.get("model_select")
        v = sel.value if sel is not None else None
        return v if v and not v.startswith("(") else None

    def sync_active_params():
        # Expose the model's selected sampler set for the chat generator to use.
        appstate.state.current_params = dict(sets.get(state["active"], sets[start]))

    def load_into_params(name: str, editable: bool):
        state["loading"] = True
        vals = sets.get(name, PRESETS[DEFAULT_PRESET])
        for key, s in sliders.items():
            s.value = vals[key]
            s.props(remove="readonly") if editable else s.props("readonly")
        state["loading"] = False

    def on_param_change(key: str):
        if state["mode"] == "edit" and not state["loading"] and state["editing"] in sets:
            sets[state["editing"]][key] = sliders[key].value

    def apply_view(name: str):
        # Called when the model panel's "Parameter set" dropdown changes (view mode).
        state["active"] = name
        sync_active_params()
        if state["mode"] == "view":
            load_into_params(name, editable=False)
            params_header.refresh()
    bridge["apply"] = apply_view

    def pset_options():
        # Dropdown labels: mark the current model's default with a trailing
        # asterisk (keys stay clean, so it shows only in the dropdown).
        d = model_defaults.get(current_model())
        return {n: (f"{n} *" if n == d else n) for n in order}

    def make_default():
        sel = bridge.get("param_select")
        m = current_model()
        if not m:
            ui.notify("Select or load a model first", type="warning")
            return
        if sel is not None:
            model_defaults[m] = sel.value
            persist()
            sel.set_options(pset_options(), value=sel.value)
            ui.notify(f"'{sel.value}' is now the default for {m}")

    def use_default():
        sel = bridge.get("param_select")
        d = model_defaults.get(current_model())
        if sel is not None and d in order:
            sel.value = d
    bridge["pset_options"] = pset_options
    bridge["make_default"] = make_default
    bridge["use_default"] = use_default

    def enter_edit():
        state["mode"] = "edit"
        # Start editing whatever set is currently selected for the model.
        state["editing"] = state["active"]
        strip.classes(add="tg-edit")
        load_into_params(state["editing"], editable=True)
        params_header.refresh()
        sets_list.refresh()

    def exit_edit():
        state["mode"] = "view"
        # Keep the last-selected set as the model's selected set.
        if state["editing"] in sets:
            state["active"] = state["editing"]
        sync_active_params()
        persist()  # flush any slider edits made in this session
        strip.classes(remove="tg-edit")
        sel = bridge.get("param_select")
        if sel is not None:
            sel.set_options(pset_options(), value=state["active"])  # reflect adds/deletes + selection
        load_into_params(state["active"], editable=False)
        params_header.refresh()

    def select_set(name: str):
        state["editing"] = name
        load_into_params(name, editable=True)
        params_header.refresh()
        sets_list.refresh()

    def new_set():
        i = 1
        while f"Set {i}" in sets:
            i += 1
        name = f"Set {i}"
        sets[name] = dict(PRESETS[DEFAULT_PRESET])
        order.append(name)
        persist()
        select_set(name)

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
        for m, s in model_defaults.items():
            if s == old:
                model_defaults[m] = new_name
        state["editing"] = new_name
        persist()
        sets_list.refresh()
        return True

    def delete_set():
        name = state["editing"]
        if name in sets:
            del sets[name]
            order.remove(name)
            for m in [m for m, s in model_defaults.items() if s == name]:
                del model_defaults[m]
            state["editing"] = order[0] if order else None
            if state["editing"]:
                load_into_params(state["editing"], editable=True)
            persist()
            params_header.refresh()
            sets_list.refresh()

    with ui.dialog() as confirm_dialog, ui.card().classes("p-5 gap-3"):
        confirm_label = ui.label().classes("text-sm")
        with ui.row().classes("w-full justify-end gap-2"):
            ui.button("Cancel", on_click=confirm_dialog.close).props("flat")
            ui.button("Delete", icon="delete",
                      on_click=lambda: (confirm_dialog.close(), delete_set())) \
                .props("color=negative unelevated")

    def ask_delete():
        if state["editing"]:
            confirm_label.text = (
                f"Delete parameter set “{state['editing']}”? This can't be undone.")
            confirm_dialog.open()

    @ui.refreshable
    def params_header():
        if state["mode"] == "view":
            ui.label("Parameter set").classes("text-xs text-muted uppercase tracking-wide")
            ui.label(state["active"]).classes("text-2xl font-semibold")
            ui.button("Edit parameter sets", icon="edit", on_click=enter_edit) \
                .props("color=primary unelevated").classes("mt-1")
        else:
            ui.label("Editing parameter set").classes("text-xs text-muted uppercase tracking-wide")
            name_input = ui.input(value=state["editing"] or "") \
                .props('filled dense input-style="font-size:1.4rem;font-weight:600"') \
                .classes("w-full tg-field")

            def commit_rename():
                if not rename_set(name_input.value):
                    name_input.value = state["editing"] or ""  # revert invalid/dup

            name_input.on("blur", commit_rename)
            name_input.on("keydown.enter", commit_rename)

            def save_now():
                persist()
                ui.notify(f"Saved '{state['editing']}'", type="positive")

            ui.button("Save", icon="save", on_click=save_now) \
                .props("color=positive unelevated").classes("mt-1")

    @ui.refreshable
    def sets_list():
        with ui.list().classes("w-full gap-1"):
            for name in order:
                item = ui.item(on_click=lambda n=name: select_set(n)).classes("tg-nav-item w-full")
                if name == state["editing"]:
                    item.classes("tg-active")
                with item:
                    with ui.item_section().props("avatar"):
                        ui.icon("tune")
                    with ui.item_section():
                        ui.label(name).classes("font-medium")

    def _params_panel():
        with ui.card().classes("w-full h-full p-5 gap-3 overflow-auto"):
            with ui.column().classes("w-full gap-1 items-start mb-2"):
                params_header()
            for key, label, minv, maxv, step, dec in PARAMS:
                sliders[key] = _slider_row(label, minv, maxv, step,
                                           PRESETS[DEFAULT_PRESET][key], dec,
                                           on_change=lambda _e, k=key: on_param_change(k))
                sliders[key].props("readonly")
            with ui.expansion("Advanced samplers", icon="tune").classes("w-full"):
                ui.label(
                    "typical_p, TFS, mirostat, DRY, XTC, penalties … will live here."
                ).classes("text-xs text-muted")

    def _sets_panel():
        with ui.card().classes("w-full h-full p-4 gap-2 overflow-hidden flex flex-col"):
            with ui.row().classes("w-full items-center justify-between"):
                ui.label("Parameter sets").classes("text-lg font-semibold")
                ui.button(icon="close", on_click=exit_edit).props("flat round dense") \
                    .tooltip("Done editing")
            with ui.row().classes("w-full gap-2 no-wrap"):
                ui.button("New", icon="add", on_click=new_set) \
                    .props("color=positive unelevated").classes("flex-1")
                ui.button(icon="delete", on_click=ask_delete) \
                    .props("color=negative unelevated").tooltip("Delete selected set")
            with ui.scroll_area().classes("flex-1 w-full rounded-lg p-1") \
                    .style("background: rgba(52,97,140,0.08)"):
                sets_list()

    # ── Filmstrip inside a clipped viewport. Panel widths as fractions of the
    #    125%-wide strip: Model 2/5, Params 2/5, SetsList 1/5. ──────────────────
    with ui.element("div").classes("w-full overflow-hidden") \
            .style("height: calc(100vh - 7rem)"):
        strip = ui.row().classes("tg-strip h-full no-wrap gap-0").style("width: 125%")
        with strip:
            with ui.element("div").classes("w-2/5 h-full px-3"):
                _model_card(bridge)
            with ui.element("div").classes("w-2/5 h-full px-3"):
                _params_panel()
            with ui.element("div").classes("w-1/5 h-full px-3"):
                _sets_panel()

    sync_active_params()  # publish the initial selection's sampler values
