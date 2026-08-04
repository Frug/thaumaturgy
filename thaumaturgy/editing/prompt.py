"""Turning a job and a span index into chat messages.

Layout matters more than wording here. Surrounding text goes in an earlier turn
and the passage arrives alone in the final one; spliced into a single context
blob the model starts reproducing at the top of the blob instead of at the
passage.
"""

from dataclasses import dataclass

from thaumaturgy.editing import spans as spans_mod
from thaumaturgy.editing.models import Job
from thaumaturgy.editing.validator import flatten

EDGE_WORDS = 8  # quoted back to pin the ends of the passage


@dataclass(frozen=True)
class _Window:
    """Span indices either side of the target that go in as context."""

    first: int
    last: int


class PromptBuilder:
    """Builds the messages for one span of one job."""

    def __init__(self, job: Job, supports_system_role: bool = True):
        self.job = job
        self.supports_system_role = supports_system_role

    def _window(self, index: int) -> _Window:
        """`overlap` tokens of corrected text behind, original ahead.

        Wider windows bury the passage in similar prose and the model wanders
        into it, so this stays centred on the span and goes no further.
        """
        spans = self.job.spans
        overlap = self.job.budgets.overlap if self.job.budgets else 0
        start = index
        used = 0
        while start > 0:
            size = spans_mod.est_tokens(spans[start - 1].current_text())
            if used + size > overlap:
                break
            start -= 1
            used += size
        end = index
        used = 0
        while end < len(spans) - 1:
            size = spans_mod.est_tokens(spans[end + 1].original)
            if used + size > overlap:
                break
            end += 1
            used += size
        return _Window(start, end)

    def context(self, index: int) -> tuple[str, str]:
        spans = self.job.spans
        w = self._window(index)
        before = "".join(s.current_text() for s in spans[w.first:index])
        after = "".join(s.original for s in spans[index + 1:w.last + 1])
        return before, after

    def messages(self, index: int, nudge: str = "") -> list[dict]:
        target = self.job.spans[index].original
        instr = self.job.instructions
        before, after = self.context(index)

        # The passage's own opening and closing words turn "the entire passage"
        # into checkable anchors. Without them the model decides for itself
        # where the interesting part starts and stops, and returns only that.
        words = flatten(target).split()
        instruction = (instr.passage_instruction
                       .replace("{first_words}", " ".join(words[:EDGE_WORDS]))
                       .replace("{last_words}", " ".join(words[-EDGE_WORDS:])))
        if nudge.strip():
            instruction = f"{instruction}\n\nAdditional instruction: {nudge.strip()}".strip()
        if "{passage}" in instruction:
            ask = instruction.replace("{passage}", target)
        else:
            ask = f"{instruction}\n\n{target}" if instruction.strip() else target

        turns: list[dict] = []
        framing = instr.context_framing
        context = ""
        if (before.strip() or after.strip()) and framing.strip():
            context = framing.replace("{before}", before).replace("{after}", after)
        primed = instr.primed_reply if instr.prime_reply else ""
        if context and primed.strip():
            turns += [{"role": "user", "content": context},
                      {"role": "assistant", "content": primed}]
        elif context:
            # Chat templates require alternating roles; merge into one user turn.
            ask = f"{context}\n\n{ask}"
        turns.append({"role": "user", "content": ask})

        system = instr.system_prompt.strip()
        if not system:
            return turns
        if self.supports_system_role:
            return [{"role": "system", "content": system}, *turns]
        turns[0]["content"] = f"{system}\n\n{turns[0]['content']}"
        return turns
