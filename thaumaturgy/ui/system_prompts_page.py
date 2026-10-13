"""System prompts page: manage reusable high-level LLM instructions."""

from nicegui import ui

from thaumaturgy import store


def render():
    """Build the System Prompts page inside the current layout container."""
    prompts = store.list_system_prompts()
    state = {"selected": 0 if prompts else None}
    guard = {"loading": False}
    pending_delete = {"index": None}
    fields: dict[str, ui.element] = {}

    def current() -> dict | None:
        index = state["selected"]
        return prompts[index] if index is not None else None

    def load_fields():
        guard["loading"] = True
        system_prompt = current() or {"name": "", "prompt_text": ""}
        fields["name"].value = system_prompt["name"]
        fields["prompt_text"].value = system_prompt["prompt_text"]
        fields["editor"].set_visibility(state["selected"] is not None)
        fields["empty"].set_visibility(state["selected"] is None)
        guard["loading"] = False

    def select(index: int):
        state["selected"] = index
        load_fields()
        prompt_list.refresh()

    def unique_new_name() -> str:
        existing = {item["name"] for item in prompts}
        base, name, number = "New system prompt", "New system prompt", 2
        while name in existing:
            name = f"{base} {number}"
            number += 1
        return name

    def add_prompt():
        item = {"name": unique_new_name(), "prompt_text": "", "_file": None}
        store.save_system_prompt(item)
        prompts.append(item)
        select(len(prompts) - 1)

    def ask_delete_prompt():
        index = state["selected"]
        if index is None:
            return
        pending_delete["index"] = index
        name = prompts[index]["name"] or "(unnamed)"
        delete_label.text = f'Delete system prompt "{name}"?'
        delete_dialog.open()

    def delete_pending_prompt():
        index = pending_delete["index"]
        pending_delete["index"] = None
        if index is None or not 0 <= index < len(prompts):
            return
        store.delete_system_prompt(prompts[index])
        del prompts[index]
        state["selected"] = min(index, len(prompts) - 1) if prompts else None
        load_fields()
        prompt_list.refresh()

    def write_back(key: str):
        item = current()
        if guard["loading"] or item is None:
            return
        item[key] = fields[key].value
        if key == "name":
            prompt_list.refresh()

    def save_current():
        item = current()
        if item is None:
            return
        store.save_system_prompt(item)
        prompt_list.refresh()
        ui.notify(f"Saved {item['name']}")

    @ui.refreshable
    def prompt_list():
        if not prompts:
            ui.label("No system prompts yet — click New.").classes(
                "text-muted text-sm p-3")
            return
        with ui.list().classes("w-full"):
            for index, item in enumerate(prompts):
                row = ui.item(on_click=lambda index=index: select(index)).classes(
                    "tg-nav-item w-full")
                if index == state["selected"]:
                    row.classes("tg-active")
                with row:
                    with ui.item_section().props("avatar"):
                        ui.icon("text_snippet")
                    with ui.item_section():
                        ui.label(item["name"] or "(unnamed)").classes("font-medium")

    with ui.dialog() as delete_dialog, ui.card().classes("p-5 gap-3") \
            .style("width:420px;max-width:92vw"):
        delete_label = ui.label().classes("text-sm leading-relaxed")
        ui.label("Chats using it will fall back to the first available prompt.") \
            .classes("text-xs text-muted leading-snug")
        with ui.row().classes("w-full justify-end gap-2"):
            ui.button("Cancel", on_click=delete_dialog.close).props("flat")
            ui.button("Delete", icon="delete",
                      on_click=lambda: (delete_dialog.close(),
                                        delete_pending_prompt())) \
                .props("color=negative unelevated")

    with ui.row().classes("w-full gap-6 no-wrap").style(
            "height: calc(100vh - 7rem)"):
        with ui.column().classes("h-full w-64 gap-2 no-wrap"):
            with ui.row().classes("w-full gap-2 no-wrap"):
                ui.button("New", icon="add", on_click=add_prompt) \
                    .props("color=positive unelevated").classes("flex-1")
                ui.button(icon="delete", on_click=ask_delete_prompt) \
                    .props("color=negative unelevated") \
                    .tooltip("Delete selected system prompt")
            with ui.scroll_area().classes("flex-1 w-full min-h-0 tg-list-shell"):
                prompt_list()

        with ui.card().classes("h-full flex-1 p-5 gap-3 overflow-auto"):
            empty = ui.column().classes(
                "w-full h-full items-center justify-center gap-2")
            with empty:
                ui.icon("text_snippet").classes("text-5xl text-muted")
                ui.label("Select a system prompt, or create a new one.").classes(
                    "text-muted")
            fields["empty"] = empty

            editor = ui.column().classes("w-full gap-3")
            with editor:
                fields["name"] = ui.input(
                    "Name", on_change=lambda: write_back("name")) \
                    .classes("w-full tg-field").props("filled")
                fields["prompt_text"] = ui.textarea(
                    "Prompt Text", on_change=lambda: write_back("prompt_text")) \
                    .classes("w-full tg-field") \
                    .props('filled input-style="height:420px"')
                with ui.row().classes("w-full mt-2"):
                    ui.button("Save system prompt", icon="save",
                              on_click=save_current) \
                        .props("color=positive unelevated")
            fields["editor"] = editor

    load_fields()
