"""Application chrome — header and left navigation drawer."""

from nicegui import app, ui

NAV: list[tuple[str, str, str]] = [
    ("Chat", "/", "chat"),
    ("Scenarios", "/scenarios", "edit_note"),
    ("Model", "/model", "memory"),
    ("Settings", "/settings", "settings"),
]


def _dark_mode() -> ui.dark_mode:
    """Dark/light preference, persisted per browser via app.storage.user."""
    return ui.dark_mode(value=app.storage.user.get("dark", True))


def render_header() -> None:
    """Render the top header with brand mark and theme toggle."""
    dark = _dark_mode()

    with ui.header().classes("tg-header items-center justify-between px-4 py-2"):
        with ui.row().classes("items-center gap-2"):
            ui.icon("auto_fix_high").classes("text-2xl text-primary")
            ui.label("thaumaturgy").classes("text-lg font-semibold tracking-wide")

        def toggle_theme():
            dark.value = not dark.value
            app.storage.user["dark"] = dark.value

        ui.button(icon="dark_mode", on_click=toggle_theme).props("flat round")


def render_left_nav(active_route: str) -> None:
    """Render the left drawer with nav items; highlight the active route."""
    with ui.left_drawer().classes("tg-drawer p-3 gap-1").props("width=210"):
        for label, route, icon in NAV:
            classes = "tg-nav-item w-full"
            if route == active_route:
                classes += " tg-active"
            item = ui.item(on_click=lambda r=route: ui.navigate.to(r)).classes(classes)
            with item:
                with ui.item_section().props("avatar"):
                    ui.icon(icon)
                with ui.item_section():
                    ui.label(label).classes("font-medium")
