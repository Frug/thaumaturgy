"""thaumaturgy entrypoint — application shell (header + nav drawer) and page routes."""

import os
from collections.abc import Callable
from contextlib import contextmanager

from nicegui import ui

from thaumaturgy import engine, theme
from thaumaturgy.ui import (chat_page, editing_page, model_page, nav,
                            scenarios_page, settings_page)

# Clean up a llama-server orphaned by a previous (reloaded) instance. Done from
# the entrypoint, not on importing engine: a test or a script that merely
# imports the package would otherwise kill whatever model is loaded.
engine.reap_stale()

ui.add_head_html(theme.head_html(), shared=True)


@contextmanager
def layout(active_route: str, pad: str = "p-2"):
    ui.colors(**theme.COLORS)
    nav.render_header()
    nav.render_left_nav(active_route)

    # Full width; each page controls its own inner max-width.
    with ui.column().classes(f"w-full {pad} gap-4") as content:
        yield content


def layout_page(route: str, pad: str = "p-2") -> Callable[[Callable[[], None]], None]:
    """Register a NiceGUI page at `route`, wrapping its body in the shared layout."""
    def decorator(render: Callable[[], None]) -> None:
        @ui.page(route)
        def _route() -> None:
            with layout(route, pad=pad):
                render()
    return decorator


@layout_page("/")
def page_chat():
    chat_page.render()


@layout_page("/scenarios")
def page_scenarios():
    scenarios_page.render()


@layout_page("/editing")
def page_editing():
    editing_page.render()


@layout_page("/model")
def page_model():
    model_page.render()


@layout_page("/settings")
def page_settings():
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
    through an installed entry point, so reload is always off here. Run the
    module form (`uv run python -m thaumaturgy.main`) for hot reload."""
    _launch(reload=False)


# NiceGUI's reloader re-imports this module as __mp_main__, and `python -m
# thaumaturgy.main` runs it as __main__ — both start with reload enabled.
if __name__ in {"__main__", "__mp_main__"}:
    main()
