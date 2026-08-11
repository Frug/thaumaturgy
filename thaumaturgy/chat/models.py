"""Chat data model.

Imports nothing from engine or store: a Chat can be built and inspected without
a loaded model or a disk.
"""

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum

from thaumaturgy.chat import reply


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


# finish_limit value for the token cap, as opposed to "context".
TOKEN_CAP_LIMIT = "max_new_tokens"


@dataclass
class Message:
    role: Role
    name: str = ""
    text: str = ""
    reasoning: str = ""
    model: str | None = None
    finish_reason: str | None = None
    finish_limit: str = TOKEN_CAP_LIMIT
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
        """Forget how the last run ended; used when a reply is hand-edited."""
        self.finish_reason = None
        self.finish_limit = TOKEN_CAP_LIMIT
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
            finish_limit=d.get("finish_limit", TOKEN_CAP_LIMIT),
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


def fingerprint(messages: list[Message]) -> str:
    """Identify a run of messages by content, so an edit to one is detectable."""
    h = hashlib.sha1()
    for m in messages:
        h.update(f"{m.role}\x1f{m.text}\x1e".encode())
    return h.hexdigest()


@dataclass
class Summary:
    """A recap standing in for the first `covers` messages of a chat.

    The messages themselves are never touched; this only changes what the model
    is sent. `fingerprint` is what they said when the recap was written, so a
    later edit to one of them retires the recap instead of silently misreporting
    it.
    """

    text: str = ""
    covers: int = 0
    fingerprint: str = ""
    tokens: int = 0
    model: str | None = None
    created: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> "Summary":
        return cls(
            text=d.get("text") or "", covers=int(d.get("covers", 0)),
            fingerprint=d.get("fingerprint") or "",
            tokens=int(d.get("tokens", 0)), model=d.get("model"),
            created=float(d.get("created", 0.0)),
        )

    def to_dict(self) -> dict:
        return {"text": self.text, "covers": self.covers,
                "fingerprint": self.fingerprint, "tokens": self.tokens,
                "model": self.model, "created": self.created}


@dataclass
class Chat:
    id: str
    scenario: str | None = None
    model: str | None = None
    title: str = "New chat"
    # Set by a rename; stops the title being derived from the first message.
    title_custom: bool = False
    created: float = 0.0
    updated: float = 0.0
    messages: list[Message] = field(default_factory=list)
    summaries: list[Summary] = field(default_factory=list)

    def append(self, message: Message) -> Message:
        self.messages.append(message)
        return message

    def active_summary(self) -> Summary | None:
        """The widest recap that still matches the messages it stands for.

        Newest first: re-compaction appends, so the last record covers the most.
        One invalidated by an edit falls back to an older, narrower one rather
        than to no recap at all.
        """
        for s in reversed(self.summaries):
            if s.covers <= len(self.messages) and s.text.strip() \
                    and s.fingerprint == fingerprint(self.messages[:s.covers]):
                return s
        return None

    def latest_assistant_index(self) -> int | None:
        """Index of the final assistant reply, when it can be redone.

        Only the last message qualifies, and only once the user has said
        something; an opening line has nothing to regenerate from.
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
            title_custom=bool(d.get("title_custom")),
            created=float(d.get("created", 0.0)),
            updated=float(d.get("updated", 0.0)),
            messages=[Message.from_dict(m) for m in (d.get("messages") or [])],
            summaries=[Summary.from_dict(s) for s in (d.get("summaries") or [])],
        )

    def to_dict(self) -> dict:
        out = {
            "id": self.id, "scenario": self.scenario, "model": self.model,
            "title": self.title, "created": self.created, "updated": self.updated,
            "messages": [m.to_dict() for m in self.messages],
        }
        if self.title_custom:
            out["title_custom"] = True
        if self.summaries:
            out["summaries"] = [s.to_dict() for s in self.summaries]
        return out


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
