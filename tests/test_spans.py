"""Splitting a document into spans.

Spans must concatenate back to the source exactly; a violation silently
corrupts a document.
"""

import random

import pytest

from thaumaturgy.editing import spans

SAMPLES = {
    "prose": ("The quick brown fox jumps over the lazy dog. It was a fine "
              "morning, crisp and clear.\n\nShe said \"we should go now!\" and "
              "left. He didn't follow.\n\nLater — much later — the rain came.\n"),
    "no-punctuation": "word " * 500,
    "one-long-sentence": "a" * 5000,
    "empty": "",
    "single-char": "x",
    "crlf": "Line one.\r\nLine two.\r\n\r\nLine three.\r\n",
    "unicode": "Héllo wörld… “quoted!” Ünicode ✓. Ünd nöch mehr Text hier.\n" * 20,
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


@pytest.mark.parametrize("target", (16, 200))
@pytest.mark.parametrize("name", SAMPLES)
def test_spans_reassemble_into_the_source(name, target):
    text = SAMPLES[name]
    parts = spans.divide(text, target)
    assert "".join(text[s:e] for s, e in parts) == text


@pytest.mark.parametrize("name", SAMPLES)
def test_spans_tile_the_source(name):
    text = SAMPLES[name]
    parts = spans.divide(text, 64)
    assert all(parts[i][1] == parts[i + 1][0] for i in range(len(parts) - 1))
    assert not parts or (parts[0][0], parts[-1][1]) == (0, len(text))
    assert all(e > s for s, e in parts)


@pytest.mark.parametrize("name", SAMPLES)
def test_spans_stay_within_budget(name):
    worst = max((e - s for s, e in spans.divide(SAMPLES[name], 64)), default=0)
    assert worst <= spans.budget_chars(64)


@pytest.mark.parametrize("target", (16, 100))
def test_spans_never_open_mid_sentence(target):
    # A span opening on a fragment makes the model complete it, duplicating
    # those words at the join.
    text = SAMPLES["hard-wrapped"]
    ends = [text[s:e].rstrip() for s, e in spans.divide(text, target)[:-1]]
    assert all(not e or e.endswith(SENTENCE_END) for e in ends), ends


@pytest.mark.parametrize("target", (16, 200))
@pytest.mark.parametrize("name", SAMPLES)
def test_tail_span_is_not_a_runt(name, target):
    # A lone trailing sentence costs a whole generation, and the model treats
    # it as a writing prompt.
    parts = spans.divide(SAMPLES[name], target)
    if len(parts) < 2:
        pytest.skip("one span")
    tail = parts[-1][1] - parts[-1][0]
    assert tail >= max(spans.MIN_SPAN_CHARS, spans.budget_chars(target) // 4)


def test_halving_preserves_the_text():
    text = SAMPLES["prose"] * 4
    parts = spans.halve(text)
    assert parts and len(parts) >= 2
    assert "".join(text[s:e] for s, e in parts) == text


def test_halving_terminates():
    # A model that truncates forever must not split forever.
    assert spans.halve("word " * 12) is None
    piece = SAMPLES["prose"] * 4
    for _ in range(200):
        got = spans.halve(piece)
        if got is None:
            break
        piece = piece[got[0][0]:got[0][1]]
    assert spans.halve(piece) is None
