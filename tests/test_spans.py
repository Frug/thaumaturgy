"""Splitting a document into spans.

The invariant that matters most: spans concatenate back to the source exactly.
Everything downstream assumes it, and a violation silently corrupts a document.
"""

import random

from tests.harness import check, section
from thaumaturgy.editing import spans

SAMPLES = {
    "prose": ("The quick brown fox jumps over the lazy dog. It was a fine "
              "morning, crisp and clear.\n\nShe said \"we should go now!\" and "
              "left. He didn't follow.\n\nLater — much later — the rain came.\n"),
    "no-punctuation": "word " * 500,
    "one-long-sentence": "a" * 5000,
    "newlines-only": "\n\n\n\n\n",
    "empty": "",
    "single-char": "x",
    "trailing-space": "Hello world.   ",
    "crlf": "Line one.\r\nLine two.\r\n\r\nLine three.\r\n",
    "unicode": "Héllo wörld… “quoted!” Ünicode ✓. Ünd nöch mehr Text hier.\n" * 20,
    "no-trailing-newline": "First sentence. Second sentence",
    "hard-wrapped": ("The village sat in a valley that received little sun. Its "
                     "houses leaned\ntoward each other like old men sharing a "
                     "secret, and its single road\nwound up toward the pass.\n"),
    "markdown": ("# Heading\n\nIntro text that wraps across\nseveral lines.\n\n"
                 "- first bullet\n- second bullet\n\n1. one\n2. two\n\nClosing.\n"),
}

random.seed(1)
SAMPLES["random"] = "".join(
    random.choice(["word ", "foo. ", "bar!\n", "baz?\n\n", "x", " ", "\n"])
    for _ in range(4000))

SENTENCE_END = tuple(".!?\"'’”)]")


def run() -> None:
    section("spans concatenate back to the source")
    for name, text in SAMPLES.items():
        for target in (16, 64, 200):
            parts = spans.divide(text, target)
            rebuilt = "".join(text[s:e] for s, e in parts)
            check(f"{name} @{target}t reassembles", rebuilt == text,
                  f"({len(rebuilt)} vs {len(text)} chars)")

    section("spans are contiguous, ordered and non-empty")
    for name, text in SAMPLES.items():
        parts = spans.divide(text, 64)
        contiguous = all(parts[i][1] == parts[i + 1][0] for i in range(len(parts) - 1))
        covers = not parts or (parts[0][0] == 0 and parts[-1][1] == len(text))
        check(f"{name} contiguous", contiguous and covers)
        check(f"{name} no zero-length", all(e > s for s, e in parts))

    section("spans stay within budget")
    for name, text in SAMPLES.items():
        budget = spans.budget_chars(64)
        worst = max((e - s for s, e in spans.divide(text, 64)), default=0)
        check(f"{name} max {worst} <= {budget}", worst <= budget)

    section("spans never open mid-sentence")
    # A span starting on a fragment makes the model complete it from context and
    # duplicate those words at the join.
    for target in (8, 16, 40, 100):
        parts = spans.divide(SAMPLES["hard-wrapped"], target)
        bad = [SAMPLES["hard-wrapped"][s:e].rstrip()[-40:]
               for (s, e), _ in zip(parts, parts[1:])
               if SAMPLES["hard-wrapped"][s:e].rstrip()
               and not SAMPLES["hard-wrapped"][s:e].rstrip().endswith(SENTENCE_END)]
        check(f"@{target}t hard-wrapped prose keeps sentences whole", not bad,
              f"after {bad[:1]}")

    section("markdown blocks start their own spans")
    parts = spans.divide(SAMPLES["markdown"], 12)
    starts = [SAMPLES["markdown"][s:e].lstrip()[:2] for s, e in parts]
    check("a bullet or heading begins a span",
          any(t.startswith(("- ", "1.", "2.", "# ")) for t in starts), f"{starts}")

    section("runt tails are folded into their neighbour")
    # Packing leaves one whenever the text divides unevenly. A lone sentence
    # costs a whole generation and the model treats it as a writing prompt.
    for name, text in SAMPLES.items():
        for target in (16, 64, 200):
            parts = spans.divide(text, target)
            if len(parts) < 2:
                continue
            tail = parts[-1][1] - parts[-1][0]
            floor = max(spans.MIN_SPAN_CHARS, spans.budget_chars(target) // 4)
            check(f"{name} @{target}t tail {tail} not a runt", tail >= floor
                  or len(parts) == 1, f"(floor {floor})")

    section("halving terminates and preserves text")
    text = SAMPLES["prose"] * 4
    check("halving a tiny span refuses", spans.halve("x") is None)
    check("halving below the floor refuses", spans.halve("word " * 12) is None)
    parts = spans.halve(text)
    check("halving returns at least two parts", parts is not None and len(parts) >= 2)
    check("halving preserves the text",
          "".join(text[s:e] for s, e in parts) == text)
    # A model that truncates forever must not split forever.
    piece = text
    for _ in range(200):
        got = spans.halve(piece)
        if got is None:
            break
        piece = piece[got[0][0]:got[0][1]]
    check("repeated halving terminates", spans.halve(piece) is None)

    section("token estimate")
    check("empty is zero", spans.est_tokens("") == 0)
    check("scales with length", spans.est_tokens("x" * 400) == 100)
