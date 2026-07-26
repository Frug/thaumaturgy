"""Run every test module. `make test`, or `uv run python -m tests`.

Deliberately dependency-free: the project has no test framework and these
checks are plain assertions over pure functions, so a runner is all it needs.
"""

import sys

from tests import harness, test_job, test_prompt, test_spans, test_validator

MODULES = (test_spans, test_validator, test_prompt, test_job)


def main() -> int:
    for module in MODULES:
        print(f"\n─── {module.__name__} ───")
        module.run()
    return harness.report()


if __name__ == "__main__":
    sys.exit(main())
