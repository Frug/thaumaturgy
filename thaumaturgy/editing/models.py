"""The editing data model.

Imports nothing from engine or store: a Job can be built and exercised without
a loaded model or a disk.
"""

from dataclasses import dataclass, field, replace
from enum import StrEnum

from thaumaturgy.editing import spans as spans_mod

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful copy editor. Fix grammar, spelling, and punctuation. "
    "Preserve the author's voice, meaning, and paragraph structure. Do not "
    "add, remove, or reorder content."
)

# Wrapper text around each passage — editable per job, competes with the system
# prompt. Placeholders: {passage}, {before}, {after}, {first_words},
# {last_words}. Missing ones are fine: the passage is appended and context is
# dropped.
DEFAULT_PASSAGE_INSTRUCTION = (
    "Output only the corrected form of the passage below. Reproduce it in "
    "full: your output must begin with \"{first_words}\" and end with "
    "\"{last_words}\". No commentary, no labels, and none of the surrounding "
    "text."
)

DEFAULT_CONTEXT_FRAMING = (
    "I will give you a passage to edit. First, for reference only, here is the "
    "text that surrounds it in the document.\n\n"
    "--- text before the passage ---\n{before}\n\n"
    "--- text after the passage ---\n{after}"
)

# Spoken as the model. Off by default — attributed words steer in ways the
# system prompt cannot see.
DEFAULT_PRIMED_REPLY = (
    "Understood. I have read the surrounding text for reference only and will "
    "not reproduce any of it. Send me the passage to edit."
)


class Status(StrEnum):
    """StrEnum so persisted job files stay readable."""

    PENDING = "pending"
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    ORIGINAL = "original"
    FLAGGED = "flagged"


DECIDED = (Status.ACCEPTED, Status.ORIGINAL)
UNDECIDED = (Status.PENDING, Status.PROPOSED, Status.FLAGGED)


def _as_str(value, fallback: str) -> str:
    """Present-but-empty is meaningful: the author cleared our wording."""
    return value if isinstance(value, str) else fallback


def _as_num(value, cast, fallback):
    try:
        return cast(value)
    except (TypeError, ValueError):
        return fallback


@dataclass
class Instructions:
    """Task wording — the part worth carrying to the next document."""

    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    passage_instruction: str = DEFAULT_PASSAGE_INSTRUCTION
    context_framing: str = DEFAULT_CONTEXT_FRAMING
    primed_reply: str = DEFAULT_PRIMED_REPLY
    prime_reply: bool = False

    @classmethod
    def from_dict(cls, vals) -> "Instructions":
        src = vals if isinstance(vals, dict) else {}
        base = cls()
        return cls(
            system_prompt=_as_str(src.get("system_prompt"), base.system_prompt),
            passage_instruction=_as_str(src.get("passage_instruction"),
                                        base.passage_instruction),
            context_framing=_as_str(src.get("context_framing"), base.context_framing),
            primed_reply=_as_str(src.get("primed_reply"), base.primed_reply),
            prime_reply=bool(src.get("prime_reply", base.prime_reply)),
        )

    def to_dict(self) -> dict:
        return {
            "system_prompt": self.system_prompt,
            "passage_instruction": self.passage_instruction,
            "context_framing": self.context_framing,
            "primed_reply": self.primed_reply,
            "prime_reply": self.prime_reply,
        }


@dataclass
class Settings:
    """Numeric tuning — belongs to a document's size and the loaded model."""

    max_new_tokens: int = 700
    temperature: float = 0.2
    top_p: float = 0.95
    top_k: int = 40
    min_p: float = 0.05
    repetition_penalty: float = 1.05
    response_buffer: int = 150   # how far under the reply cap to size a span
    # Neighbouring prose is the main cause of wandering; raise carefully.
    overlap_pct: float = 0.0
    auto_accept_clean: bool = False
    allow_deletions: bool = False

    @classmethod
    def from_dict(cls, vals) -> "Settings":
        src = vals if isinstance(vals, dict) else {}
        b = cls()
        out = cls(
            max_new_tokens=_as_num(src.get("max_new_tokens", b.max_new_tokens),
                                   int, b.max_new_tokens),
            temperature=_as_num(src.get("temperature", b.temperature), float, b.temperature),
            top_p=_as_num(src.get("top_p", b.top_p), float, b.top_p),
            top_k=_as_num(src.get("top_k", b.top_k), int, b.top_k),
            min_p=_as_num(src.get("min_p", b.min_p), float, b.min_p),
            repetition_penalty=_as_num(src.get("repetition_penalty", b.repetition_penalty),
                                       float, b.repetition_penalty),
            response_buffer=_as_num(src.get("response_buffer", b.response_buffer),
                                    int, b.response_buffer),
            overlap_pct=_as_num(src.get("overlap_pct", b.overlap_pct), float, b.overlap_pct),
            auto_accept_clean=bool(src.get("auto_accept_clean", b.auto_accept_clean)),
            allow_deletions=bool(src.get("allow_deletions", b.allow_deletions)),
        )
        return replace(
            out,
            overlap_pct=min(0.45, max(0.0, out.overlap_pct)),
            response_buffer=max(0, min(out.response_buffer, out.max_new_tokens - 64)),
        )

    def to_dict(self) -> dict:
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "repetition_penalty": self.repetition_penalty,
            "response_buffer": self.response_buffer,
            "overlap_pct": self.overlap_pct,
            "auto_accept_clean": self.auto_accept_clean,
            "allow_deletions": self.allow_deletions,
        }

    def sampler_params(self) -> dict:
        """The subset engine.stream_chat consumes."""
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "repetition_penalty": self.repetition_penalty,
        }


@dataclass(frozen=True)
class Budgets:
    """Token allowances for one job, derived from the loaded server."""

    span_target: int
    available: int
    overlap: int

    PROMPT_OVERHEAD = 256   # chat template, restated span, instruction text
    SAFETY_RESERVE = 256

    @classmethod
    def derive(cls, settings: Settings, context_limit: int | None,
               system_tokens: int = 0, reasoning_budget: int = 0) -> "Budgets":
        max_new = settings.max_new_tokens
        span_target = max(64, max_new - settings.response_buffer)
        reserve = max_new + system_tokens + cls.SAFETY_RESERVE + max(0, reasoning_budget)
        usable = max(1024, (context_limit or 8192) - reserve)
        # The span appears twice: marked in place, and restated at the tail.
        available = max(512, usable - 2 * span_target - cls.PROMPT_OVERHEAD)
        return cls(span_target=span_target, available=available,
                   overlap=int(available * settings.overlap_pct))


@dataclass
class Span:
    start: int
    end: int
    original: str
    rewritten: str = ""
    status: Status = Status.PENDING
    finish_reason: str | None = None
    flags: list[str] = field(default_factory=list)
    attempts: int = 0

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def is_decided(self) -> bool:
        return self.status in DECIDED

    def current_text(self) -> str:
        """What this span contributes to the document as it stands."""
        return self.rewritten if self.status == Status.ACCEPTED else self.original

    @classmethod
    def from_dict(cls, d: dict) -> "Span":
        return cls(
            start=int(d["start"]), end=int(d["end"]),
            original=d.get("original", ""), rewritten=d.get("rewritten", ""),
            status=Status(d.get("status", Status.PENDING)),
            finish_reason=d.get("finish_reason"),
            flags=list(d.get("flags") or []),
            attempts=int(d.get("attempts", 0)),
        )

    def to_dict(self) -> dict:
        return {
            "start": self.start, "end": self.end, "original": self.original,
            "rewritten": self.rewritten, "status": str(self.status),
            "finish_reason": self.finish_reason, "flags": list(self.flags),
            "attempts": self.attempts,
        }


@dataclass
class Job:
    id: str
    title: str
    source_text: str
    instructions: Instructions
    settings: Settings
    spans: list[Span] = field(default_factory=list)
    budgets: Budgets | None = None
    model: str | None = None
    created: float = 0.0
    updated: float = 0.0

    # ── spans ────────────────────────────────────────────────────────────────
    def divide(self) -> None:
        """Build the span list. Only on first open; spans are then persisted."""
        # Without a loaded server to size against, the settings alone still give
        # a span target — the reply cap less the response buffer.
        budgets = self.budgets or Budgets.derive(self.settings, None)
        self.spans = [Span(s, e, self.source_text[s:e])
                      for s, e in spans_mod.divide(self.source_text,
                                                   budgets.span_target)]

    def split(self, index: int) -> bool:
        """Replace one oversize span with two. False once too small to divide.

        Refuses a decided span: the replacements start empty and pending, which
        would silently discard an accepted rewrite.
        """
        span = self.spans[index]
        if span.is_decided:
            return False
        parts = spans_mod.halve(span.original)
        if parts is None:
            return False
        base = span.start
        self.spans[index:index + 1] = [
            Span(base + s, base + e, self.source_text[base + s:base + e])
            for s, e in parts]
        return True

    # ── progress ─────────────────────────────────────────────────────────────
    def next_undecided(self, after: int = -1) -> int | None:
        for i in range(after + 1, len(self.spans)):
            if not self.spans[i].is_decided:
                return i
        return None

    def progress(self) -> tuple[int, int]:
        """Decided progress in source characters, not span count.

        Truncated spans are halved and retried, so the span count climbs
        mid-run and a span-based bar slides backwards.
        """
        if not self.spans:
            return 0, 0
        done = sum(s.length for s in self.spans if s.is_decided)
        return done, self.spans[-1].end - self.spans[0].start

    def percent(self) -> int:
        done, total = self.progress()
        return round(done / total * 100) if total else 0

    def assemble(self) -> str:
        """The document as edited so far; undecided spans fall back to original."""
        return "".join(s.current_text() for s in self.spans)

    def decide(self, index: int, status: Status, text: str | None = None) -> None:
        span = self.spans[index]
        if text is not None:
            span.rewritten = reattach_edges(span.original, text)
        span.status = status

    # ── serialisation ────────────────────────────────────────────────────────
    @classmethod
    def from_dict(cls, d: dict) -> "Job":
        settings = Settings.from_dict(d.get("settings"))
        instructions = Instructions.from_dict(
            {**(d.get("settings") or {}), "system_prompt": d.get("system_prompt")})
        return cls(
            id=d["id"], title=d.get("title") or "Untitled",
            source_text=d.get("source_text", ""),
            instructions=instructions, settings=settings,
            spans=[Span.from_dict(s) for s in (d.get("spans") or [])],
            model=d.get("model"),
            created=float(d.get("created", 0.0)), updated=float(d.get("updated", 0.0)),
        )

    def to_dict(self) -> dict:
        # Wrapper text rides in settings so a hand-edited job file keeps one
        # block of prompt knobs rather than two.
        settings = {**self.settings.to_dict(), **self.instructions.to_dict()}
        settings.pop("system_prompt", None)
        return {
            "id": self.id, "title": self.title, "model": self.model,
            "created": self.created, "updated": self.updated,
            "system_prompt": self.instructions.system_prompt,
            "settings": settings,
            "source_text": self.source_text,
            "spans": [s.to_dict() for s in self.spans],
        }


def reattach_edges(original: str, text: str) -> str:
    """Restore the span's leading/trailing whitespace from the source.

    Models drop boundary whitespace; a lost newline at a join welds paragraphs.
    """
    lead = original[:len(original) - len(original.lstrip())]
    trail = original[len(original.rstrip()):]
    return lead + text.strip() + trail
