"""Model-tracked state variables.

The model narrates freely, but it's a poor ledger: numbers drift, especially
across long contexts and after compaction blurs the past. So the authoritative
values live here in code, not in the token stream. Each turn the model emits a
hidden ``<STATE>{...}</STATE>`` block of *changed* variables; we parse it out,
merge it into ``chat["state"]``, and re-inject the current values into the system
prompt next turn. A bad generation can't corrupt the stored value, and the model
reads the truth each turn instead of re-deriving it from history.

This module is pure: parse a block, merge a dict, format for prompt/display. The
wiring (stream stripping, injection, UI) lives in the chat page.
"""

import json
import re

# The model wraps a JSON object of changed variables in this block, at the very
# end of its reply. Non-greedy up to the first closing tag so nested JSON objects
# survive intact.
_STATE_RE = re.compile(r"<STATE>\s*(.*?)\s*</STATE>", re.DOTALL | re.IGNORECASE)
# A block that's been opened but not yet closed — i.e. still streaming in, or
# malformed. Hidden from the user but never parsed.
_STATE_OPEN_RE = re.compile(r"<STATE>.*\Z", re.DOTALL | re.IGNORECASE)

# Injected into the system prompt so the model knows the protocol. What to track
# is left to the scenario's own prose; this only defines the mechanism.
PROTOCOL_INSTRUCTION = (
    "State tracking: keep track of the meaningful variables this scenario implies "
    "— things like health, resources or inventory counts, location, time, or "
    "relationship levels. Whenever one changes, end your reply with a single "
    "hidden block `<STATE>{...}</STATE>` holding a JSON object of only the changed "
    "variables and their new values, for example "
    "`<STATE>{\"hp\": 12, \"gold\": 43, \"location\": \"tavern\"}</STATE>`. Set a "
    "variable to null to drop it. This block is hidden from the player: never "
    "mention it, and write nothing after it. Omit it entirely when nothing changed."
)


def visible(text: str) -> str:
    """Strip state blocks (and any trailing unclosed one) for display."""
    text = _STATE_RE.sub("", text)
    text = _STATE_OPEN_RE.sub("", text)
    return text.strip()


def extract(text: str) -> tuple[str, dict | None]:
    """Return ``(clean_text, updates)`` — the reply minus state blocks, plus the
    merged updates parsed from them (``None`` when there were none/unparseable)."""
    updates: dict = {}
    for match in _STATE_RE.finditer(text):
        try:
            obj = json.loads(match.group(1))
        except ValueError:
            continue
        if isinstance(obj, dict):
            updates.update(obj)
    return visible(text), (updates or None)


def apply(state: dict, updates: dict) -> dict:
    """Merge ``updates`` into a copy of ``state``; a null value drops the key.

    Unmentioned keys are left untouched, so the model forgetting to restate a
    variable never loses it.
    """
    out = dict(state)
    for key, value in updates.items():
        if value is None:
            out.pop(key, None)
        else:
            out[key] = value
    return out


def format_value(value) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def format_for_prompt(state: dict) -> str:
    """The authoritative current values, injected into the system prompt."""
    if not state:
        return ""
    lines = [f"{key}: {format_value(value)}" for key, value in state.items()]
    return "Current state:\n" + "\n".join(lines)
