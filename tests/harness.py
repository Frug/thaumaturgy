"""A check function and a tally. No framework, no dependencies."""

FAILURES: list[str] = []
PASSED = [0]


def check(name: str, condition, detail: str = "") -> None:
    if condition:
        PASSED[0] += 1
    else:
        FAILURES.append(f"{name} {detail}".rstrip())
        print(f"  FAIL {name} {detail}".rstrip())


def section(title: str) -> None:
    print(f"\n== {title} ==")


def report() -> int:
    print()
    if FAILURES:
        print(f"{len(FAILURES)} failed, {PASSED[0]} passed")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print(f"All {PASSED[0]} checks passed.")
    return 0
