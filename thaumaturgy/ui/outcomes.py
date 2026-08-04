"""Rendering a service Outcome for the user."""

from nicegui import ui

# Both services name these steps the same, and StrEnum members compare as
# their values, so one table serves every page.
_KIND = {"blocked": "warning", "error": "negative"}


def notify(outcome) -> None:
    """Toast whatever the service reported, if it said anything."""
    if not outcome.message:
        return
    kind = _KIND.get(str(outcome.step))
    if kind:
        ui.notify(outcome.message, type=kind)
    else:
        ui.notify(outcome.message)
