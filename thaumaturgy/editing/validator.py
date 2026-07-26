"""Checks on a model's rewrite of one passage.

Pure text in, reasons out — no job dicts, no spans, no settings. Every check
here answers "should a human look at this before it lands in the document",
and each exists because a real rewrite failed that way while every other
check read it as clean.
"""

from difflib import SequenceMatcher

MAX_RATIO = 2.0
MIN_RATIO = 0.5
# When deletions are allowed, only near-total gutting is flagged.
MIN_RATIO_DELETING = 0.1

_RATIO_FLOOR = 40
_BLEED_WINDOW = 60
_ALIGN_BLOCK = 20  # shortest shared run treated as real alignment

# Closing words of the rewrite must fall this far through the original.
MIN_END_COVERAGE = 0.8

# Full-length replies are assumed to have covered the span regardless of edges.
MIN_END_LENGTH = 0.9

# Opening words may fall at most this far into the original.
MAX_START_OFFSET = 0.1

# Share of *output* that must trace to the original. Asymmetric on purpose:
# deletion scores ~1.0, invention ~0.15. Summaries pass here but fail MIN_RATIO.
MIN_DRAWN_FROM_SOURCE = 0.6


def flatten(text: str) -> str:
    return " ".join(text.split())


def _aligned_blocks(a: str, b: str) -> list:
    return [bl for bl in SequenceMatcher(None, a, b).get_matching_blocks()
            if bl.size >= _ALIGN_BLOCK]


def drawn_from_source(original: str, text: str) -> float:
    """Share of `text` that traces back to `original`, 0..1.

    Asymmetric: deletion leaves this at 1.0 (remaining chars still from source);
    invented prose drives it down. Symmetric similarity would punish deletion.
    """
    a, b = flatten(original), flatten(text)
    if not b:
        return 1.0
    matched = sum(block.size for block in
                  SequenceMatcher(None, a, b).get_matching_blocks())
    return matched / len(b)


def reaches_end(original: str, text: str) -> bool:
    """Whether the rewrite carries through to the end of the span.

    Deletion and truncation both come back short; alignment of the output's
    closing words against the original separates them. Use matching blocks
    (monotonic) rather than searching for the closing words — search finds the
    last occurrence, which in repetitive prose sits at the end regardless.
    """
    a, b = flatten(original), flatten(text)
    if len(a) < _BLEED_WINDOW or len(b) < _BLEED_WINDOW:
        return True
    # Full-length reply that merely reworded the close is not truncation.
    if len(b) >= len(a) * MIN_END_LENGTH:
        return True
    blocks = _aligned_blocks(a, b)
    if not blocks:
        return True  # nothing of the author's survives → invention's to flag
    last = blocks[-1]
    return (last.a + last.size) / len(a) >= MIN_END_COVERAGE


def starts_at_start(original: str, text: str) -> bool:
    """Whether the rewrite opens where the passage opens.

    Mirror of reaches_end. Skipping ahead (e.g. to one speaker's lines) looks
    clean to every other check — no invention, ending lines up, length ok.
    """
    a, b = flatten(original), flatten(text)
    if len(a) < _BLEED_WINDOW or len(b) < _BLEED_WINDOW:
        return True
    if len(b) >= len(a) * MIN_END_LENGTH:
        return True
    blocks = _aligned_blocks(a, b)
    if not blocks:
        return True  # nothing of the author's survives → invention's to flag
    return blocks[0].a <= len(a) * MAX_START_OFFSET


def bleeds(edge: str, text: str, at_start: bool) -> bool:
    """True when the rewrite runs on into neighbouring context.

    Anchor on the output's boundary run, find it in the neighbour, and confirm
    overlap continues to the join. Searching for neighbour text anywhere would
    flag every span of a repetitive document. Whitespace flattened so reflow
    still counts.
    """
    edge, text = flatten(edge), flatten(text)
    if len(edge) < _BLEED_WINDOW or len(text) < _BLEED_WINDOW:
        return False
    probe = text[:_BLEED_WINDOW] if at_start else text[-_BLEED_WINDOW:]
    pos = edge.find(probe)
    while pos != -1:
        if at_start and edge[pos:] == text[:len(edge) - pos]:
            return True
        if not at_start and edge[:pos + _BLEED_WINDOW] == text[-(pos + _BLEED_WINDOW):]:
            return True
        pos = edge.find(probe, pos + 1)
    return False


def check(original: str, text: str, finish_reason: str | None, *,
          before: str = "", after: str = "",
          allow_deletions: bool = False) -> list[str]:
    """Reasons this rewrite should not be auto-accepted."""
    flags = []
    if finish_reason == "length":
        flags.append("truncated")
    if not text.strip():
        flags.append("empty")
        return flags
    # Internal paragraph breaks are the model's to keep; lost ones merge
    # paragraphs. Skip when deletions are allowed (content removal takes blanks).
    if not allow_deletions and text.count("\n\n") < original.count("\n\n"):
        flags.append("lost-break")
    if drawn_from_source(original, text) < MIN_DRAWN_FROM_SOURCE:
        flags.append("invented")
    if not reaches_end(original, text):
        flags.append("stops-short")
    if not starts_at_start(original, text):
        flags.append("starts-late")
    # Short spans: one added word can swing the ratio past the threshold alone.
    if len(original) >= _RATIO_FLOOR:
        floor = MIN_RATIO_DELETING if allow_deletions else MIN_RATIO
        ratio = len(text) / len(original)
        if ratio < floor or ratio > MAX_RATIO:
            flags.append("length-ratio")
    # Only when the rewrite grew: bleeding adds neighbour text. Gate also avoids
    # flagging periodic prose where a bleed reads like correct continuation.
    if len(flatten(text)) - len(flatten(original)) >= _BLEED_WINDOW:
        if bleeds(before, text, at_start=True) or bleeds(after, text, at_start=False):
            flags.append("context-bleed")
    return flags
