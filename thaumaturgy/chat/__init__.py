"""Conversations with the loaded model.

The data types and service have no NiceGUI dependency. Pages create their own
service while sharing the process-wide runtime for live generations.
"""

from thaumaturgy.chat.models import Chat, Message, Role, Scenario
from thaumaturgy.chat.service import ChatService, Step, chat_runtime

__all__ = [
    "Chat",
    "ChatService",
    "Message",
    "Role",
    "Scenario",
    "Step",
    "chat_runtime",
]
