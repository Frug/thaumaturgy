"""Settings page: app preferences (data dir, theme, flags).

Content is stubbed for now; the module exists so the route matches the other
pages (chat / scenarios / model) with a proper render() entrypoint.
"""

from nicegui import ui


def render() -> None:
    """Build the Settings page inside the current layout container."""
    with ui.card().classes("w-full max-w-3xl mx-auto items-center p-10 gap-2"):
        ui.label("Settings").classes("text-2xl font-semibold")
        ui.label("App settings (data dir, theme, flags) will live here.").classes(
            "text-muted")
