"""thaumaturgy entrypoint — application shell (header + nav drawer) and page routes."""

import os
from contextlib import contextmanager

from nicegui import app, ui

from thaumaturgy import theme
from thaumaturgy.ui import chat_page, model_page, scenarios_page

ui.add_head_html(theme.head_html(), shared=True)

NAV = [
    ("Chat", "/", "chat"),
    ("Scenarios", "/scenarios", "edit_note"),
    ("Model", "/model", "memory"),
    ("Settings", "/settings", "settings"),
]


def _dark_mode() -> ui.dark_mode:
    """Dark/light preference, persisted per browser via app.storage.user."""
    return ui.dark_mode(value=app.storage.user.get("dark", True))


@contextmanager
def layout(active_route: str, pad: str = "p-6"):
    """Shared chrome (header + nav drawer) wrapping each page's content."""
    theme.apply_colors()
    dark = _dark_mode()

    with ui.header().classes("tg-header items-center justify-between px-4 py-2"):
        with ui.row().classes("items-center gap-2"):
            ui.icon("auto_fix_high").classes("text-2xl text-primary")
            ui.label("thaumaturgy").classes("text-lg font-semibold tracking-wide")

        def toggle_theme():
            dark.value = not dark.value
            app.storage.user["dark"] = dark.value

        ui.button(icon="dark_mode", on_click=toggle_theme).props("flat round")

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

    # Full width; each page controls its own inner max-width.
    with ui.column().classes(f"w-full {pad} gap-4") as content:
        yield content


def _placeholder(title: str, subtitle: str):
    with ui.card().classes("w-full max-w-3xl mx-auto items-center p-10 gap-2"):
        ui.label(title).classes("text-2xl font-semibold")
        ui.label(subtitle).classes("text-muted")


@ui.page("/")
def page_chat():
    with layout("/", pad="p-2"):
        chat_page.render()


@ui.page("/scenarios")
def page_scenarios():
    with layout("/scenarios", pad="p-2"):
        scenarios_page.render()


@ui.page("/model")
def page_model():
    with layout("/model", pad="p-2"):
        model_page.render()


@ui.page("/settings")
def page_settings():
    with layout("/settings"):
        _placeholder("Settings", "App settings (data dir, theme, flags) will live here.")


def _launch(reload: bool):
    ui.run(
        title="thaumaturgy",
        port=int(os.environ.get("THAUM_PORT", "8080")),
        storage_secret="thaumaturgy-dev",  # enables app.storage.user (theme persistence)
        reload=reload,
        show=False,
    )


def main():
    """Module entry (`python -m thaumaturgy.main`): hot-reload on by default."""
    _launch(reload=os.environ.get("THAUM_NO_RELOAD") != "1")


def cli():
    """Console-script entry (`uv run thaumaturgy`). NiceGUI's reloader can't work
    through an installed entry point, so reload is always off here — run the
    module form (`uv run python -m thaumaturgy.main`) for hot reload."""
    _launch(reload=False)


# NiceGUI's reloader re-imports this module as __mp_main__, and `python -m
# thaumaturgy.main` runs it as __main__ — both start with reload enabled.
if __name__ in {"__main__", "__mp_main__"}:
    main()
