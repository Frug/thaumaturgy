"""Summarizing context compaction.

When a chat grows close to the model's context window, this folds the older
turns into a running summary so the model keeps getting a version that fits,
while the user still sees the full transcript. The summary is stored on the chat
(`chat["compaction"]`) so it survives reloads and is only regenerated when the
context creeps back toward the limit — not on every message.

The plugin is pure with respect to the UI: the caller supplies closures for
building the outgoing message list, counting tokens, and summarizing text. It
never touches NiceGUI or the engine directly.
"""

from dataclasses import dataclass

from thaumaturgy import plugins

# Start compacting once the outgoing context reaches this fraction of the budget
# (i.e. "within 10% of the limit").
TRIGGER_RATIO = 0.90

# Never fold the most recent N messages into the summary. Compacting something
# the user just wrote — or the reply they're reading — would be jarring, so the
# tail stays verbatim.
RECENT_KEEP = 5

# Summary sizing (tokens). The summary is capped so that scenario + summary +
# recent tail is guaranteed to land under budget: MARGIN absorbs tokenization
# drift, MIN is the least room worth summarizing into (below it we give up), MAX
# keeps summaries bounded on large-context models.
SUMMARY_MARGIN = 64
MIN_SUMMARY_TOKENS = 128
MAX_SUMMARY_TOKENS = 1024

STATUS_UNCHANGED = "unchanged"
STATUS_COMPACTED = "compacted"
STATUS_FALLBACK = "fallback"

_COMPACTED_MSG = "Older messages were summarized to stay within the context window."
_ACTIVE_MSG = "Earlier messages are summarized for the model; you still see the full chat."
_FALLBACK_MSG = (
    "Context window is too small to fit this chat even after compacting — "
    "sending the full history without compaction."
)

# The compaction instruction (the system message). Customizable via the caller;
# this is the fallback. It stays neutral about the chat's purpose — the scenario
# is handed to the model separately, wrapped in <SCENARIO> tags, and this tells
# the model to leave that part alone and only summarize the transcript.
DEFAULT_INSTRUCTION = (
    "Summarize the conversation transcript below into a concise synopsis that "
    "preserves the important facts, characters, decisions, and unresolved "
    "threads, so it can stand in for those messages. The text inside the "
    "<SCENARIO> tags is background context — do not summarize or repeat it, only "
    "use it to understand the transcript. Output only the synopsis, with no "
    "preamble."
)


@dataclass
class CompactionResult:
    """Outcome of a compaction attempt for one send.

    `api` is the message list to actually send. `status` is one of the STATUS_*
    constants. `feedback` is a short user-facing note (or None) to show below the
    context counter.
    """

    api: list
    status: str
    feedback: str | None


def active_note(chat) -> str | None:
    """Persistent note for a chat whose history is currently summarized."""
    if (chat.get("compaction") or {}).get("summary"):
        return _ACTIVE_MSG
    return None


def _summary_prompt(scenario_prompt: str, prev_summary: str, folded: list[dict]) -> str:
    """The user message: the scenario (tagged, keep) then the transcript (compact).

    The instruction lives in the system message; this is just the material.
    """
    parts = []
    if scenario_prompt:
        parts.append(f"<SCENARIO>\n{scenario_prompt}\n</SCENARIO>")
    lines = []
    if prev_summary:
        # The running synopsis is part of what's being (re)summarized.
        lines.append(f"[Summary of earlier messages]\n{prev_summary}")
    for m in folded:
        role = "User" if m.get("role") == "user" else "Assistant"
        text = (m.get("text") or "").strip()
        if text:
            lines.append(f"{role}: {text}")
    parts.append("Transcript to summarize:\n" + "\n".join(lines))
    return "\n\n".join(parts)


def compact(chat, *, build_api, count_tokens, summarize, budget,
            scenario_prompt="", instruction=None,
            recent_keep=RECENT_KEEP, trigger_ratio=TRIGGER_RATIO):
    """Decide whether to compact `chat`, and return the messages to send.

    Args:
      chat: the chat dict (mutated in place when a new summary is stored).
      build_api: `build_api(chat, use_compaction=bool) -> list[dict]`, the
        outgoing message list. With compaction on it substitutes the stored
        summary for the folded head; off it returns the full history.
      count_tokens: `count_tokens(api) -> int`, prompt tokens for a message list.
      summarize: `summarize(instruction, user, max_tokens) -> str`, one model
        call. `instruction` is the system message, `user` the material.
      budget: prompt-token budget (context window minus generation headroom), or
        None/0 when unknown (then compaction is skipped).
      scenario_prompt: the scenario's system prompt, handed to the model wrapped
        in <SCENARIO> tags so it can frame the transcript without being summarized.
      instruction: the compaction instruction (system message); DEFAULT_INSTRUCTION
        when None.
    """
    instruction = instruction or DEFAULT_INSTRUCTION

    api = build_api(chat, use_compaction=True)
    if not budget or budget <= 0:
        return CompactionResult(api, STATUS_UNCHANGED, None)

    used = count_tokens(api)
    if used < trigger_ratio * budget:
        # Room to spare; keep the current (possibly already-compacted) context.
        note = _ACTIVE_MSG if (chat.get("compaction") or {}).get("summary") else None
        return CompactionResult(api, STATUS_UNCHANGED, note)

    messages = chat.get("messages", [])
    comp = chat.get("compaction") or {}
    covered = min(max(comp.get("covered", 0), 0), len(messages))
    fold_end = max(0, len(messages) - recent_keep)

    def give_up():
        # Send the full history uncompacted and warn. Crucially, nothing on the
        # chat has been mutated, so this doesn't spend a summary call and won't
        # keep re-triggering: same length in, same decision out.
        return CompactionResult(build_api(chat, use_compaction=False),
                                STATUS_FALLBACK, _FALLBACK_MSG)

    if fold_end <= covered:
        # The recent tail is the whole remaining history — nothing left to fold.
        if used >= budget:
            return give_up()
        note = _ACTIVE_MSG if comp.get("summary") else None
        return CompactionResult(api, STATUS_UNCHANGED, note)

    # Floor: the smallest the compacted prompt can be — scenario + the recent
    # tail, summary contributing nothing. Measured *before* summarizing, so if
    # there isn't room for a worthwhile summary we give up without spending a
    # model call or touching the chat.
    floor = count_tokens(build_api({"messages": messages[fold_end:]},
                                   use_compaction=False))
    room = budget - floor - SUMMARY_MARGIN
    if room < MIN_SUMMARY_TOKENS:
        return give_up()

    # Cap the summary to the room the tail leaves, so scenario + summary + tail
    # is under budget by construction. Fold the head (prior summary + the turns
    # that have aged past the recent tail) and commit only on a real result.
    cap = min(MAX_SUMMARY_TOKENS, room)
    try:
        new_summary = summarize(
            instruction,
            _summary_prompt(scenario_prompt, comp.get("summary", ""),
                            messages[covered:fold_end]),
            cap).strip()
    except Exception:  # noqa: BLE001 - a failed summary shouldn't break the send
        return give_up()
    if not new_summary:
        return give_up()

    chat["compaction"] = {"summary": new_summary, "covered": fold_end}
    return CompactionResult(build_api(chat, use_compaction=True),
                            STATUS_COMPACTED, _COMPACTED_MSG)


plugins.register_compactor("summarize", compact)
