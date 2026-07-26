"""Checks on a model's rewrite.

Every case here is a failure a real run produced, together with the shapes that
must NOT flag — the false positives cost more than the misses, because a
reviewer who learns to ignore the badges gains nothing from them.
"""

from tests.harness import check, section
from thaumaturgy.editing import Validator

BODY = ("Real prose that should survive the edit intact. " * 12 +
        "\n\nPAGE FURNITURE LINE\nANOTHER JUNK LINE\n\n" +
        "More real prose carrying on afterwards for a while. " * 12)

JUNK_GONE = BODY.replace("\n\nPAGE FURNITURE LINE\nANOTHER JUNK LINE\n\n", "\n\n")

# Non-repeating: in periodic prose a dropped head re-aligns to an earlier
# repeat, and a bleed reads exactly like correct continuation.
UNIQUE = "".join(f"Sentence {n} sets out topic {n} at some length here. "
                 for n in range(80))


def run() -> None:
    v = Validator()
    deleting = Validator(allow_deletions=True)

    section("a clean rewrite is silent")
    check("unchanged text", v.check(BODY, BODY, "stop") == [])
    check("typo fixed", v.check(BODY, BODY.replace("prose", "porse", 1), "stop") == [])

    section("mechanical failures")
    check("truncated", "truncated" in v.check(BODY, BODY, "length"))
    check("empty", v.check(BODY, "   ", "stop") == ["empty"])
    check("summarised", "length-ratio" in v.check(BODY, "It describes events.", "stop"))
    check("padded", "length-ratio" in v.check(BODY, BODY * 3, "stop"))

    section("invention vs deletion")
    invented = "Entirely fresh sentences bearing no relation to the source. " * 14
    check("invented prose flags", "invented" in v.check(BODY, invented, "stop"))
    check("invented flags even when deletions allowed",
          "invented" in deleting.check(BODY, invented, "stop"))
    check("pure deletion scores 1.0", v.drawn_from_source(BODY, BODY[:len(BODY) // 2]) == 1.0)
    check("junk removal scores 1.0", v.drawn_from_source(BODY, JUNK_GONE) == 1.0)
    check("typo fix stays high",
          v.drawn_from_source(BODY, BODY.replace("prose", "porse", 1)) > 0.95)
    check("invention scores low", v.drawn_from_source(BODY, invented) < 0.6)
    check("empty output does not divide by zero", v.drawn_from_source(BODY, "") == 1.0)

    section("deletion is an instruction, not a defect")
    check("stripping junk passes when allowed", not deleting.check(BODY, JUNK_GONE, "stop"))
    # ...and is flagged without the setting, which is the point of having it.
    check("stripping junk flags in copy-edit mode", v.check(BODY, JUNK_GONE, "stop"))
    check("flattened breaks flag in copy-edit mode",
          "lost-break" in v.check(BODY, BODY.replace("\n\n", " "), "stop"))
    check("flattened breaks tolerated when deleting",
          "lost-break" not in deleting.check(BODY, BODY.replace("\n\n", " "), "stop"))
    check("gutting still flags when deleting",
          "length-ratio" in deleting.check(BODY, "Short.", "stop"))

    section("stops-short: gave up, vs reworded the ending")
    check("unchanged reaches the end", v.reaches_end(BODY, BODY))
    check("junk removed mid-body reaches the end", v.reaches_end(BODY, JUNK_GONE))
    for frac in (0.40, 0.60, 0.75):
        check(f"stopping at {frac:.0%} flags", not v.reaches_end(BODY, BODY[:int(len(BODY) * frac)]))
    filler = ("She moved through the ruined field with slow deliberate care "
              "and said nothing further that evening. ")
    for frac in (0.10, 0.30, 0.50):
        cut = int(len(BODY) * (1 - frac))
        reworded = BODY[:cut] + (filler * 40)[:len(BODY) - cut]
        check(f"tail reworded {frac:.0%} at full length does not flag",
              v.reaches_end(BODY, reworded), f"(len {len(reworded)/len(BODY):.2f}x)")
    check("a full-length reply never stops short", v.reaches_end(BODY, "x" * len(BODY)))

    section("starts-late: skipped the opening")
    # Verbatim tail with the head dropped passes every other check — nothing
    # invented, ending correct, ratio inside bounds.
    check("unchanged starts at the start", v.starts_at_start(UNIQUE, UNIQUE))
    for frac in (0.25, 0.44, 0.60):
        check(f"dropping the first {frac:.0%} flags",
              not v.starts_at_start(UNIQUE, UNIQUE[int(len(UNIQUE) * frac):]))
    check("trimming a few opening words is tolerated",
          v.starts_at_start(UNIQUE, UNIQUE[int(len(UNIQUE) * 0.03):]))
    check("a full-length reply never starts late", v.starts_at_start(UNIQUE, "x" * len(UNIQUE)))
    check("stopping early is not mistaken for starting late",
          v.starts_at_start(UNIQUE, UNIQUE[:len(UNIQUE) // 2]))
    check("head-dropped verbatim tail is flagged by check",
          "starts-late" in v.check(UNIQUE, UNIQUE[int(len(UNIQUE) * 0.44):], "stop"))

    section("context bleed")
    before = UNIQUE[:600]
    target = UNIQUE[600:1400]
    check("clean rewrite does not bleed", "context-bleed" not in
          v.check(target, target, "stop", before=before, after=UNIQUE[1400:2000]))
    for bled in (80, 200, 500):
        out = before[-bled:] + target
        check(f"a {bled}-char bleed flags", "context-bleed" in
              v.check(target, out, "stop", before=before, after=UNIQUE[1400:2000]))
    check("reflowed bleed still flags", "context-bleed" in v.check(
        target, " ".join((before[-200:] + target).split()), "stop",
        before=before, after=UNIQUE[1400:2000]))
    check("sub-threshold bleed does not flag", "context-bleed" not in
          v.check(target, before[-10:] + target, "stop", before=before))

    section("repetitive prose does not false-positive")
    # A periodic document is the pathological case: a bleed reads identically to
    # correct continuation, so the checks must not fire on unchanged text.
    for name, text in (("unicode", "Héllo wörld… ✓ Ünd nöch mehr.\n" * 60),
                       ("no-punctuation", "word " * 400),
                       ("prose", BODY * 3)):
        flags = v.check(text, text, "stop")
        check(f"{name}: unchanged text is silent", not flags, f"{flags}")

    section("thresholds are per-instance")
    strict = Validator(min_drawn_from_source=0.999)
    check("a tuned validator is stricter",
          "invented" in strict.check(BODY, BODY.replace("prose", "porse", 1), "stop"))
    check("the default is not", "invented" not in
          v.check(BODY, BODY.replace("prose", "porse", 1), "stop"))
