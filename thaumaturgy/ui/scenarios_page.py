"""Scenarios page — manage conversation setups."""

from nicegui import ui

from thaumaturgy import store


def render():
    """Build the Scenarios page inside the current layout container."""
    scenarios = store.list_scenarios()
    state = {"selected": 0 if scenarios else None}
    guard = {"loading": False}
    fields: dict[str, ui.element] = {}

    def current() -> dict | None:
        i = state["selected"]
        return scenarios[i] if i is not None else None

    def load_fields():
        guard["loading"] = True
        scenario = current() or {"name": "", "context": "", "opening_text": ""}
        fields["name"].value = scenario["name"]
        fields["context"].value = scenario["context"]
        fields["opening_text"].value = scenario["opening_text"]
        fields["editor"].set_visibility(state["selected"] is not None)
        fields["empty"].set_visibility(state["selected"] is None)
        guard["loading"] = False

    def select(i: int):
        state["selected"] = i
        load_fields()
        scenario_list.refresh()

    def unique_new_name() -> str:
        existing = {s["name"] for s in scenarios}
        base, name, i = "New scenario", "New scenario", 2
        while name in existing:
            name = f"{base} {i}"
            i += 1
        return name

    def add_scenario():
        s = {
            "name": unique_new_name(),
            "context": "",
            "opening_text": "",
            "_file": None,
        }
        store.save_scenario(s)
        scenarios.append(s)
        select(len(scenarios) - 1)

    def delete_scenario():
        i = state["selected"]
        if i is None:
            return
        store.delete_scenario(scenarios[i])
        del scenarios[i]
        state["selected"] = min(i, len(scenarios) - 1) if scenarios else None
        load_fields()
        scenario_list.refresh()

    def write_back(key: str):
        scenario = current()
        if guard["loading"] or scenario is None:
            return
        scenario[key] = fields[key].value
        if key == "name":
            scenario_list.refresh()

    def save_current():
        scenario = current()
        if scenario is None:
            return
        store.save_scenario(scenario)
        scenario_list.refresh()
        ui.notify(f"Saved {scenario['name']}")

    @ui.refreshable
    def scenario_list():
        if not scenarios:
            ui.label("No scenarios yet — click New.").classes("text-muted text-sm p-3")
            return
        with ui.list().classes("w-full"):
            for i, s in enumerate(scenarios):
                item = ui.item(on_click=lambda i=i: select(i)).classes("tg-nav-item w-full")
                if i == state["selected"]:
                    item.classes("tg-active")
                with item:
                    with ui.item_section().props("avatar"):
                        ui.icon("edit_note")
                    with ui.item_section():
                        ui.label(s["name"] or "(unnamed)").classes("font-medium")

    with ui.row().classes("w-full gap-6 no-wrap").style("height: calc(100vh - 7rem)"):
        with ui.column().classes("h-full w-64 gap-2 no-wrap"):
            with ui.row().classes("w-full gap-2 no-wrap"):
                ui.button("New", icon="add", on_click=add_scenario) \
                    .props("color=positive unelevated").classes("flex-1")
                ui.button(icon="delete", on_click=delete_scenario) \
                    .props("color=negative unelevated") \
                    .tooltip("Delete selected scenario")
            with ui.scroll_area().classes("flex-1 w-full min-h-0 tg-list-shell"):
                scenario_list()

        with ui.card().classes("h-full flex-1 p-5 gap-3 overflow-auto"):
            empty = ui.column().classes("w-full h-full items-center justify-center gap-2")
            with empty:
                ui.icon("edit_note").classes("text-5xl text-muted")
                ui.label("Select a scenario, or create a new one.").classes("text-muted")
            fields["empty"] = empty

            editor = ui.column().classes("w-full gap-3")
            with editor:
                fields["name"] = ui.input("Name", on_change=lambda: write_back("name")) \
                    .classes("w-full tg-field").props("filled")
                fields["context"] = ui.textarea(
                    "Scenario context", on_change=lambda: write_back("context")) \
                    .classes("w-full tg-field").props('filled input-style="height:240px"')
                fields["opening_text"] = ui.textarea(
                    "Opening text", on_change=lambda: write_back("opening_text")) \
                    .classes("w-full tg-field").props('filled input-style="height:180px"')

                with ui.row().classes("w-full mt-2"):
                    ui.button("Save scenario", icon="save", on_click=save_current) \
                        .props("color=positive unelevated")
            fields["editor"] = editor

    load_fields()
