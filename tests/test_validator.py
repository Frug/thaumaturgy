"""Checks on a model's rewrite.

Every case is a failure a real run produced, paired with the shapes that must
NOT flag — a reviewer who learns to ignore the badges gains nothing from them.
"""

import pytest

from thaumaturgy.editing import Validator

BODY = ("Real prose that should survive the edit intact. " * 12 +
        "\n\nPAGE FURNITURE LINE\nANOTHER JUNK LINE\n\n" +
        "More real prose carrying on afterwards for a while. " * 12)

JUNK_GONE = BODY.replace("\n\nPAGE FURNITURE LINE\nANOTHER JUNK LINE\n\n", "\n\n")

# Non-repeating: in periodic prose a dropped head re-aligns to an earlier repeat.
UNIQUE = "".join(f"Sentence {n} sets out topic {n} at some length here. "
                 for n in range(80))


@pytest.fixture
def v():
    return Validator()


@pytest.fixture
def deleting():
    return Validator(allow_deletions=True)


def test_a_clean_rewrite_is_silent(v):
    assert v.check(BODY, BODY, "stop") == []
    assert v.check(BODY, BODY.replace("prose", "porse", 1), "stop") == []


def test_mechanical_failures(v):
    assert "truncated" in v.check(BODY, BODY, "length")
    assert v.check(BODY, "   ", "stop") == ["empty"]
    assert "length-ratio" in v.check(BODY, "It describes events.", "stop")
    assert "length-ratio" in v.check(BODY, BODY * 3, "stop")


def test_invented_prose_flags_in_either_mode(v, deleting):
    invented = "Entirely fresh sentences bearing no relation to the source. " * 14
    assert "invented" in v.check(BODY, invented, "stop")
    assert "invented" in deleting.check(BODY, invented, "stop")
    assert v.drawn_from_source(BODY, invented) < 0.6


def test_deleted_text_still_counts_as_drawn_from_the_source(v):
    assert v.drawn_from_source(BODY, JUNK_GONE) == 1.0
    assert v.drawn_from_source(BODY, "") == 1.0  # no division by zero


def test_deletion_is_an_instruction_not_a_defect(v, deleting):
    assert not deleting.check(BODY, JUNK_GONE, "stop")
    assert v.check(BODY, JUNK_GONE, "stop")
    assert "lost-break" in v.check(BODY, BODY.replace("\n\n", " "), "stop")
    assert "lost-break" not in deleting.check(BODY, BODY.replace("\n\n", " "), "stop")
    assert "length-ratio" in deleting.check(BODY, "Short.", "stop")


@pytest.mark.parametrize("frac", (0.40, 0.75))
def test_stopping_short_flags(v, frac):
    assert not v.reaches_end(BODY, BODY[:int(len(BODY) * frac)])


def test_a_reworded_ending_is_not_mistaken_for_stopping_short(v):
    assert v.reaches_end(BODY, BODY)
    assert v.reaches_end(BODY, JUNK_GONE)
    filler = ("She moved through the ruined field with slow deliberate care "
              "and said nothing further that evening. ")
    cut = int(len(BODY) * 0.7)
    assert v.reaches_end(BODY, BODY[:cut] + (filler * 40)[:len(BODY) - cut])


def test_starting_late_flags(v):
    # A verbatim tail with the head dropped passes every other check.
    late = UNIQUE[int(len(UNIQUE) * 0.44):]
    assert not v.starts_at_start(UNIQUE, late)
    assert "starts-late" in v.check(UNIQUE, late, "stop")


def test_starting_late_ignores_a_trimmed_opening_and_an_early_stop(v):
    assert v.starts_at_start(UNIQUE, UNIQUE)
    assert v.starts_at_start(UNIQUE, UNIQUE[int(len(UNIQUE) * 0.03):])
    assert v.starts_at_start(UNIQUE, UNIQUE[:len(UNIQUE) // 2])


@pytest.mark.parametrize("bled", (80, 500))
def test_context_bleed_flags(v, bled):
    before, target = UNIQUE[:600], UNIQUE[600:1400]
    assert "context-bleed" in v.check(target, before[-bled:] + target, "stop",
                                      before=before, after=UNIQUE[1400:2000])


def test_context_bleed_survives_reflowing_but_needs_a_real_overlap(v):
    before, target, after = UNIQUE[:600], UNIQUE[600:1400], UNIQUE[1400:2000]
    assert "context-bleed" not in v.check(target, target, "stop",
                                          before=before, after=after)
    reflowed = " ".join((before[-200:] + target).split())
    assert "context-bleed" in v.check(target, reflowed, "stop",
                                      before=before, after=after)
    assert "context-bleed" not in v.check(target, before[-10:] + target, "stop",
                                          before=before)


# The pathological case: a bleed reads identically to correct continuation.
@pytest.mark.parametrize("text", ("Héllo wörld… ✓ Ünd nöch mehr.\n" * 60,
                                  "word " * 400,
                                  BODY * 3),
                         ids=("unicode", "no-punctuation", "prose"))
def test_repetitive_prose_does_not_false_positive(v, text):
    assert v.check(text, text, "stop") == []
