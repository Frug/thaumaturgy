"""Span-by-span document editing.

`editor` is the one entry point: it owns the open job and the in-flight run,
and every call returns an Outcome the caller renders. The models are here for
constructing and inspecting a job; everything else is an implementation detail
of the submodules. No NiceGUI; the page is presentation only.
"""

from thaumaturgy.editing.models import Instructions, Job, Settings, Span, Status
from thaumaturgy.editing.service import Step, editor
from thaumaturgy.editing.validator import Validator

__all__ = [
    "Instructions",
    "Job",
    "Settings",
    "Span",
    "Status",
    "Step",
    "Validator",
    "editor",
]
