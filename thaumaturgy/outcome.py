"""What a service call did, for its page to render.

Shared by the chat and editing services; the `step` values are each service's
own enum, so the page it belongs to knows how to read them.
"""

from dataclasses import dataclass
from enum import StrEnum


@dataclass(frozen=True)
class Outcome:
    step: StrEnum
    message: str = ""
