"""Chat data model.

Imports nothing from engine or store: a Chat can be built and inspected without
a loaded model or a disk.
"""

from dataclasses import dataclass, field
from enum import StrEnum

from thaumaturgy.chat import reply


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    role: Role
    name: str = ""
    text: str = ""
    reasoning: str = ""
    model: str | None = None
    finish_reason: str | None = None
    finish_limit: str = "max_new_tokens"
    generation_error: str | None = None

    @property
    def is_user(self) -> bool:
        return self.role is Role.USER

    def display(self) -> tuple[str, str]:
        """Visible text and reasoning, with channel markers resolved."""
        text, reasoning = self.text or "", (self.reasoning or "").strip()
        if self.role is Role.ASSISTANT:
            text, marker_reasoning = reply.split_channels(text)
            reasoning = reasoning or marker_reasoning
        return reply.promote_reasoning(text, reasoning)

    def warning(self) -> str | None:
        """Why this reply is suspect, if it is."""
        if self.generation_error:
            return f"Generation failed: {self.generation_error}"
        reason = self.finish_reason
        if not reason or reason == "stop":
            return None
        if reason == "error":
            return "Generation failed before the model finished replying."
        if reason == "length":
            if self.finish_limit == "context":
                return ("Generation stopped because the context window filled up. "
                        "Max new tokens doesn't apply while the reasoning budget "
                        "is unrestricted.")
            return "Generation stopped because Max new tokens was reached."
        return f"Generation finished with reason: {reason}."

    def clear_generation_state(self) -> None:
        """Forget how the last run ended — used when a reply is hand-edited."""
        self.finish_reason = None
        self.finish_limit = "max_new_tokens"
        self.generation_error = None
        self.reasoning = ""

    @classmethod
    def from_dict(cls, d: dict) -> "Message":
        return cls(
            role=Role(d.get("role", Role.USER)),
            name=d.get("name") or "",
            text=d.get("text") or "",
            reasoning=d.get("reasoning") or "",
            model=d.get("model"),
            finish_reason=d.get("finish_reason"),
            finish_limit=d.get("finish_limit", "max_new_tokens"),
            generation_error=d.get("generation_error"),
        )

    def to_dict(self) -> dict:
        out = {"role": str(self.role), "name": self.name, "text": self.text}
        # Optional keys stay absent when empty so hand-read chat files stay tidy.
        for key, value in (("reasoning", self.reasoning), ("model", self.model),
                           ("finish_reason", self.finish_reason),
                           ("generation_error", self.generation_error)):
            if value:
                out[key] = value
        if self.finish_reason == "length":
            out["finish_limit"] = self.finish_limit
        return out


@dataclass
class Chat:
    id: str
    scenario: str | None = None
    model: str | None = None
    title: str = "New chat"
    created: float = 0.0
    updated: float = 0.0
    messages: list[Message] = field(default_factory=list)

    def append(self, message: Message) -> Message:
        self.messages.append(message)
        return message

    def latest_assistant_index(self) -> int | None:
        """Index of the final assistant reply, when it can be redone.

        Only the last message qualifies, and only once the user has said
        something — an opening line has nothing to regenerate from.
        """
        if not self.messages or self.messages[-1].role is not Role.ASSISTANT:
            return None
        if not any(m.role is Role.USER for m in self.messages[:-1]):
            return None
        return len(self.messages) - 1

    @classmethod
    def from_dict(cls, d: dict) -> "Chat":
        return cls(
            id=d["id"], scenario=d.get("scenario"), model=d.get("model"),
            title=d.get("title") or "New chat",
            created=float(d.get("created", 0.0)),
            updated=float(d.get("updated", 0.0)),
            messages=[Message.from_dict(m) for m in (d.get("messages") or [])],
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id, "scenario": self.scenario, "model": self.model,
            "title": self.title, "created": self.created, "updated": self.updated,
            "messages": [m.to_dict() for m in self.messages],
        }


@dataclass(frozen=True)
class Scenario:
    """A conversation setup: who the model is playing, and how it opens."""

    name: str
    context: str = ""
    opening_text: str = ""
    file: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "Scenario":
        return cls(name=d.get("name") or "", context=d.get("context") or "",
                   opening_text=d.get("opening_text") or "", file=d.get("_file"))
