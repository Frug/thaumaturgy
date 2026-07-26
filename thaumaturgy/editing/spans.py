"""Cutting a document into spans small enough for one reply.

Pure text in, offsets out. Nothing here touches a model, a job, or a disk.
"""

import math
import re

# Rough bytes-per-token for packing. Spans are verified exactly before send.
CHARS_PER_TOKEN = 4

# Floor so repeated truncation can't shave a span down to nothing.
MIN_SPAN_CHARS = 40

# Boundaries land after sentence punctuation, at paragraph breaks, and before
# markdown blocks — never at a bare newline, which is a soft wrap in
# hard-wrapped prose. A span opening mid-sentence makes the model complete the
# fragment from context and duplicate those words at the join.
_BOUNDARY_RE = re.compile(
    r"[.!?][\"'’”)\]]*\s+"                    # sentence end + following gap
    r"|\n[ \t]*\n\s*"                          # blank line (paragraph break)
    r"|\n(?=[ \t]*(?:[-*+>#]|\d+[.)])[ \t])")  # before a markdown block

Offsets = list[tuple[int, int]]


def est_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def split_atoms(text: str) -> Offsets:
    """Cut `text` at sentence/line boundaries into contiguous atoms."""
    out: Offsets = []
    pos = 0
    for m in _BOUNDARY_RE.finditer(text):
        if m.end() > pos:
            out.append((pos, m.end()))
            pos = m.end()
    if pos < len(text):
        out.append((pos, len(text)))
    return out


def _hard_split(text: str, start: int, end: int, budget: int) -> Offsets:
    """Break an atom that alone exceeds the budget, preferring whitespace."""
    out: Offsets = []
    pos = start
    while end - pos > budget:
        cut = text.rfind(" ", pos + budget // 2, pos + budget)
        cut = cut + 1 if cut > pos else pos + budget
        out.append((pos, cut))
        pos = cut
    if pos < end:
        out.append((pos, end))
    return out


def pack(text: str, budget_chars: int) -> Offsets:
    """Greedily group atoms into spans of at most `budget_chars`.

    Offsets are contiguous and cover the whole text, so reassembly can never
    lose or duplicate a character.
    """
    atoms: Offsets = []
    for a_start, a_end in split_atoms(text):
        if a_end - a_start > budget_chars:
            atoms.extend(_hard_split(text, a_start, a_end, budget_chars))
        else:
            atoms.append((a_start, a_end))

    out: Offsets = []
    start = end = None
    for a_start, a_end in atoms:
        if start is None:
            start, end = a_start, a_end
        elif a_end - start > budget_chars:
            out.append((start, end))
            start, end = a_start, a_end
        else:
            end = a_end
    if start is not None:
        out.append((start, end))

    # Fold a runt tail into its neighbour. Tiny spans waste a generation and the
    # model treats the fragment as a writing prompt. Skip if merging would
    # overshoot the budget enough to risk truncation.
    if len(out) > 1:
        runt = out[-1][1] - out[-1][0]
        merged = out[-1][1] - out[-2][0]
        if runt < max(MIN_SPAN_CHARS, budget_chars // 4) and merged <= budget_chars * 1.15:
            tail = out.pop()
            out[-1] = (out[-1][0], tail[1])
    return out


def budget_chars(span_target_tokens: int) -> int:
    return max(200, span_target_tokens * CHARS_PER_TOKEN)


def divide(text: str, span_target_tokens: int) -> Offsets:
    return pack(text, budget_chars(span_target_tokens))


def halve(text: str) -> Offsets | None:
    """Split one span's text in two. None once it is too small to divide."""
    if len(text) < 2 * MIN_SPAN_CHARS:
        return None
    parts = pack(text, max(MIN_SPAN_CHARS, len(text) // 2))
    return parts if len(parts) >= 2 else None
