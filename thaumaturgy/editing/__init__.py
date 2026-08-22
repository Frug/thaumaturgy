"""Span-by-span document editing.

Pages own their selected job while the shared runtime retains live workflows
across navigation. The models are here for constructing and inspecting a job;
everything else is an implementation detail of the submodules. No NiceGUI;
the page is presentation only.
"""

from thaumaturgy.editing.models import Instructions, Job, Settings, Span, Status
from thaumaturgy.editing.service import EditingService, Step, editing_runtime
from thaumaturgy.editing.validator import Validator

__all__ = [
    "Instructions",
    "EditingService",
    "Job",
    "Settings",
    "Span",
    "Status",
    "Step",
    "Validator",
    "editing_runtime",
]
