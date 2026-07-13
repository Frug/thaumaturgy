"""Characters page — manage RP characters the LLM can play.

Left: full-height scrollable list of character names, with New / Delete above it.
Right: the selected character's editable properties (name, context, opening text).

Characters are persisted as one YAML file each under <data>/characters/. Edits
apply to the in-memory copy; "Save character" writes to disk (New persists
immediately so the character shows up in the chat page's selector).
"""

from nicegui import ui

from thaumaturgy import store


def render():
    """Build the Characters page inside the current layout container."""
    chars = store.list_characters()
    state = {"selected": 0 if chars else None}
    guard = {"loading": False}
    fields: dict[str, ui.element] = {}

    def load_fields():
        guard["loading"] = True
        i = state["selected"]
        c = chars[i] if i is not None else {"name": "", "context": "", "greeting": ""}
        fields["name"].value = c["name"]
        fields["context"].value = c["context"]
        fields["greeting"].value = c["greeting"]
        fields["editor"].set_visibility(i is not None)
        fields["empty"].set_visibility(i is None)
        guard["loading"] = False

    def select(i: int):
        state["selected"] = i
        load_fields()
        char_list.refresh()

    def unique_new_name() -> str:
        existing = {c["name"] for c in chars}
        base, name, i = "New character", "New character", 2
        while name in existing:
            name = f"{base} {i}"
            i += 1
        return name

    def add_char():
        c = {"name": unique_new_name(), "context": "", "greeting": "", "_file": None}
        store.save_character(c)  # persist so it appears in the chat selector
        chars.append(c)
        select(len(chars) - 1)

    def delete_char():
        i = state["selected"]
        if i is None:
            return
        store.delete_character(chars[i])
        del chars[i]
        state["selected"] = min(i, len(chars) - 1) if chars else None
        load_fields()
        char_list.refresh()

    def write_back(key: str):
        if guard["loading"] or state["selected"] is None:
            return
        chars[state["selected"]][key] = fields[key].value
        if key == "name":
            char_list.refresh()

    def save_current():
        i = state["selected"]
        if i is None:
            return
        store.save_character(chars[i])
        char_list.refresh()
        ui.notify(f"Saved {chars[i]['name']}")

    @ui.refreshable
    def char_list():
        if not chars:
            ui.label("No characters yet — click New.").classes("text-muted text-sm p-3")
            return
        with ui.list().classes("w-full gap-1"):
            for i, c in enumerate(chars):
                item = ui.item(on_click=lambda i=i: select(i)).classes("tg-nav-item w-full")
                if i == state["selected"]:
                    item.classes("tg-active")
                with item:
                    with ui.item_section().props("avatar"):
                        ui.icon("person")
                    with ui.item_section():
                        ui.label(c["name"] or "(unnamed)").classes("font-medium")

    with ui.row().classes("w-full gap-6 no-wrap").style("height: calc(100vh - 7rem)"):
        # ── LEFT: New/Delete + scrollable name list ──────────────────────────
        with ui.column().classes("h-full w-64 gap-2 no-wrap"):
            with ui.row().classes("w-full gap-2 no-wrap"):
                ui.button("New", icon="add", on_click=add_char) \
                    .props("color=positive unelevated").classes("flex-1")
                ui.button(icon="delete", on_click=delete_char) \
                    .props("color=negative unelevated") \
                    .tooltip("Delete selected character")
            with ui.scroll_area().classes("flex-1 w-full rounded-lg p-1") \
                    .style("background: rgba(52,97,140,0.08)"):
                char_list()

        # ── RIGHT: selected character's properties ───────────────────────────
        with ui.card().classes("h-full flex-1 p-5 gap-3 overflow-auto"):
            empty = ui.column().classes("w-full h-full items-center justify-center gap-2")
            with empty:
                ui.icon("person_outline").classes("text-5xl text-muted")
                ui.label("Select a character, or create a new one.").classes("text-muted")
            fields["empty"] = empty

            editor = ui.column().classes("w-full gap-3")
            with editor:
                fields["name"] = ui.input("Name", on_change=lambda: write_back("name")) \
                    .classes("w-full tg-field").props("filled")
                fields["context"] = ui.textarea(
                    "Context", on_change=lambda: write_back("context")) \
                    .classes("w-full tg-field").props('filled input-style="height:280px"')
                fields["greeting"] = ui.textarea(
                    "Opening text", on_change=lambda: write_back("greeting")) \
                    .classes("w-full tg-field").props('filled input-style="height:280px"')
                ui.label(
                    "Opening text is the character's first message when a new chat starts "
                    "(textgen calls this the greeting)."
                ).classes("text-xs text-muted -mt-1")
                with ui.row().classes("w-full mt-2"):
                    ui.button("Save character", icon="save", on_click=save_current) \
                        .props("color=positive unelevated")
            fields["editor"] = editor

    load_fields()
