"""Minimal plugin seam for thaumaturgy.

Right now there is exactly one extension point: **context compaction** — the
strategy that shrinks a long chat so it keeps fitting the model's context
window. Plugins register a named compactor at import time; the app asks for one
by name (falling back to the built-in "summarize").

This is intentionally small. It borrows the low-ceremony spirit of
text-generation-webui's extensions (a module exposes a well-known function and
gets called) without the transformers-era hook surface that doesn't apply to a
llama-server-over-HTTP app. New hook kinds can be added here as they're needed.
"""

import importlib
import pkgutil

# name -> compactor callable. See compaction.CompactionResult for the contract.
_COMPACTORS: dict[str, object] = {}
_discovered = False


def register_compactor(name: str, fn) -> None:
    _COMPACTORS[name] = fn


def _discover() -> None:
    """Import every plugin module in this package so it can self-register."""
    global _discovered
    if _discovered:
        return
    _discovered = True
    for mod in pkgutil.iter_modules(__path__):
        if not mod.name.startswith("_"):
            importlib.import_module(f"{__name__}.{mod.name}")


def get_compactor(name: str = "summarize"):
    """Return the named compactor, or None if it isn't registered."""
    _discover()
    return _COMPACTORS.get(name)
