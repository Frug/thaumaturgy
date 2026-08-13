"""Telling the user what happened, on a page that may not be there any more.

A long await (e.g. from loading a model, downloading one, folding a chat) outlives the
page that started it if the user navigates away meanwhile. NiceGUI resolves a
toast against that page's element tree, so once the tree is collected it raises
instead of notifying. Nothing has gone wrong when that happens: the work ran to
completion, there is just nobody left to tell, so the message goes to the
terminal instead of up the stack as an error.
"""

from nicegui import ui

# Both services name these steps the same, and StrEnum members compare as
# their values, so one table serves every page.
_KIND = {"blocked": "warning", "error": "negative"}


def toast(message: str, kind: str | None = None) -> None:
    """Show a message to whoever asked for it, if their page is still open."""
    try:
        ui.notify(message, type=kind)
    except RuntimeError:
        print(f"[thaumaturgy] {message} (page closed before it could be shown)")


def notify(outcome) -> None:
    """Toast whatever the service reported, if it said anything."""
    if outcome.message:
        toast(outcome.message, _KIND.get(str(outcome.step)))
