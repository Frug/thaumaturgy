"""Checks on a model's rewrite of one passage.

Pure text in, reasons out: no jobs, no spans, no settings. Each check exists
because a real rewrite failed that way while every other check read it as clean.
"""

from dataclasses import dataclass
from difflib import SequenceMatcher


def flatten(text: str) -> str:
    return " ".join(text.split())


@dataclass(frozen=True)
class Validator:
    """Thresholds plus the checks that use them.

    Deletion is a legitimate instruction ("strip the scraped page furniture"),
    so `allow_deletions` relaxes the length floor and the paragraph-break check
    while leaving invention and truncation caught.
    """

    allow_deletions: bool = False

    max_ratio: float = 2.0
    min_ratio: float = 0.5
    min_ratio_deleting: float = 0.1
    # Below this share of the output tracing back to the original, the model
    # wrote new prose. Asymmetric on purpose: deletion scores ~1.0, invention
    # ~0.15. A summary passes here but fails the length floor.
    min_drawn_from_source: float = 0.6
    # Closing words must fall this far through the original; opening words no
    # further than this into it.
    min_end_coverage: float = 0.8
    max_start_offset: float = 0.1
    # A full-length reply has covered the span whatever its edges align to.
    min_end_length: float = 0.9

    ratio_floor: int = 40      # below this, one added word swings the ratio
    edge_window: int = 60      # shortest run treated as a boundary probe
    align_block: int = 20      # shortest shared run treated as real alignment

    # ── measures ─────────────────────────────────────────────────────────────
    def _blocks(self, a: str, b: str) -> list:
        return [bl for bl in SequenceMatcher(None, a, b).get_matching_blocks()
                if bl.size >= self.align_block]

    def drawn_from_source(self, original: str, text: str) -> float:
        """Share of `text` that traces back to `original`, 0..1.

        Deletion leaves this at 1.0; every remaining character still came from
        the source. A symmetric similarity would punish deletion instead.
        """
        a, b = flatten(original), flatten(text)
        if not b:
            return 1.0
        matched = sum(block.size for block in
                      SequenceMatcher(None, a, b).get_matching_blocks())
        return matched / len(b)

    def reaches_end(self, original: str, text: str) -> bool:
        """Whether the rewrite carries through to the end of the span.

        Deletion and truncation both come back short; aligning the closing
        words separates them. Matching blocks are monotonic, whereas searching
        for the closing words finds their last occurrence, which in repetitive
        prose sits at the end regardless.
        """
        a, b = flatten(original), flatten(text)
        if len(a) < self.edge_window or len(b) < self.edge_window:
            return True
        if len(b) >= len(a) * self.min_end_length:
            return True
        blocks = self._blocks(a, b)
        if not blocks:
            return True  # nothing of the author's survives → invention's to flag
        last = blocks[-1]
        return (last.a + last.size) / len(a) >= self.min_end_coverage

    def starts_at_start(self, original: str, text: str) -> bool:
        """Whether the rewrite opens where the passage opens.

        Mirror of reaches_end. Skipping ahead (to one speaker's lines, say)
        looks clean to every other check: no invention, ending lines up, length
        inside the ratio.
        """
        a, b = flatten(original), flatten(text)
        if len(a) < self.edge_window or len(b) < self.edge_window:
            return True
        if len(b) >= len(a) * self.min_end_length:
            return True
        blocks = self._blocks(a, b)
        if not blocks:
            return True
        return blocks[0].a <= len(a) * self.max_start_offset

    def bleeds(self, edge: str, text: str, at_start: bool) -> bool:
        """True when the rewrite runs on into neighbouring context.

        Anchor on the output's boundary run, find it in the neighbour, and
        confirm the overlap continues to the join. Searching for neighbour text
        anywhere would flag every span of a repetitive document.
        """
        edge, text = flatten(edge), flatten(text)
        w = self.edge_window
        if len(edge) < w or len(text) < w:
            return False
        probe = text[:w] if at_start else text[-w:]
        pos = edge.find(probe)
        while pos != -1:
            if at_start and edge[pos:] == text[:len(edge) - pos]:
                return True
            if not at_start and edge[:pos + w] == text[-(pos + w):]:
                return True
            pos = edge.find(probe, pos + 1)
        return False

    # ── verdict ──────────────────────────────────────────────────────────────
    def check(self, original: str, text: str, finish_reason: str | None, *,
              before: str = "", after: str = "") -> list[str]:
        """Reasons this rewrite should not be auto-accepted."""
        flags = []
        if finish_reason == "length":
            flags.append("truncated")
        if not text.strip():
            flags.append("empty")
            return flags
        # Internal paragraph breaks are the model's to keep; losing one merges
        # paragraphs. Moot once deletions are allowed: removing content removes
        # the blank lines around it.
        if not self.allow_deletions and text.count("\n\n") < original.count("\n\n"):
            flags.append("lost-break")
        if self.drawn_from_source(original, text) < self.min_drawn_from_source:
            flags.append("invented")
        if not self.reaches_end(original, text):
            flags.append("stops-short")
        if not self.starts_at_start(original, text):
            flags.append("starts-late")
        if len(original) >= self.ratio_floor:
            floor = self.min_ratio_deleting if self.allow_deletions else self.min_ratio
            ratio = len(text) / len(original)
            if ratio < floor or ratio > self.max_ratio:
                flags.append("length-ratio")
        # Only when the rewrite grew: bleeding adds neighbour text. The gate
        # also stops periodic prose flagging, where a bleed reads like correct
        # continuation.
        if len(flatten(text)) - len(flatten(original)) >= self.edge_window:
            if self.bleeds(before, text, True) or self.bleeds(after, text, False):
                flags.append("context-bleed")
        return flags
