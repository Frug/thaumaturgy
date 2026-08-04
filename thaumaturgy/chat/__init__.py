"""Conversations with the loaded model.

`chat` is the entry point: it owns the open conversation, its scenario, and any
reply in flight, and every call returns an Outcome the caller renders. No
NiceGUI; the page is presentation only.
"""

from thaumaturgy.chat.models import Chat, Message, Role, Scenario
from thaumaturgy.chat.service import Step, chat

__all__ = [
    "Chat",
    "Message",
    "Role",
    "Scenario",
    "Step",
    "chat",
]
