"""thaumaturgy entrypoint — application shell (header + nav drawer) and page routes."""

import os
from contextlib import contextmanager

from nicegui import ui

from thaumaturgy import theme
from thaumaturgy.ui import chat_page, model_page, nav, scenarios_page, settings_page

ui.add_head_html(theme.head_html(), shared=True)


@contextmanager
def layout(active_route: str, pad: str = "p-2"):
    ui.colors(**theme.COLORS)
    nav.render_header()
    nav.render_left_nav(active_route)

    # Full width; each page controls its own inner max-width.
    with ui.column().classes(f"w-full {pad} gap-4") as content:
        yield content


@ui.page("/")
def page_chat():
    with layout("/"):
        chat_page.render()


@ui.page("/scenarios")
def page_scenarios():
    with layout("/scenarios"):
        scenarios_page.render()


@ui.page("/model")
def page_model():
    with layout("/model"):
        model_page.render()


@ui.page("/settings")
def page_settings():
    with layout("/settings"):
        settings_page.render()


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
