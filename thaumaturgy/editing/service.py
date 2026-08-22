"""The editing service: selected jobs, shared workflows, and their rules.

Every method returns an Outcome describing what happened, so the whole state
machine (retry-on-truncation, auto-accept, cursor advance) can be driven from
a script with no UI attached.

The runtime is process-wide so a browser reload cannot orphan a run. Service
instances are page-local so tabs do not change one another's selected job.
"""

import json
import time
from dataclasses import dataclass
from enum import StrEnum, auto

from thaumaturgy import appstate, engine, store
from thaumaturgy.editing import spans as spans_mod
from thaumaturgy.editing.models import (
    Budgets,
    Instructions,
    Job,
    Settings,
    Span,
    Status,
    reattach_edges,
)
from thaumaturgy.editing.prompt import PromptBuilder
from thaumaturgy.editing.runner import SpanRun
from thaumaturgy.editing.validator import Validator
from thaumaturgy.lang import en
from thaumaturgy.outcome import Outcome
from thaumaturgy.paths import log_dir


class Step(StrEnum):
    REVIEW = auto()       # a rewrite is waiting for a decision
    ADVANCED = auto()     # accepted automatically; a new run may be in flight
    SPLIT_RETRY = auto()  # too long for one reply, halved and retrying
    COMPLETE = auto()     # nothing left undecided
    CANCELLED = auto()
    ERROR = auto()
    BLOCKED = auto()      # no model loaded, or the server is busy
    IDLE = auto()         # nothing open


@dataclass
class EditingState:
    """One live document workflow, shared by pages attached to that job."""

    job: Job
    run: SpanRun | None = None
    index: int | None = None
    sent: tuple[int, list[dict]] | None = None


class EditingRuntime:
    """Process-wide editing workflows which must survive page navigation."""

    def __init__(self):
        self.states: dict[str, EditingState] = {}

    def remember(self, state: EditingState) -> EditingState:
        self.states[state.job.id] = state
        return state

    def forget(self, job_id: str) -> None:
        self.states.pop(job_id, None)

    def busy(self) -> bool:
        return any(s.run is not None and not s.run.done for s in self.states.values())

    def occupied(self, job_id: str) -> bool:
        state = self.states.get(job_id)
        return state is not None and state.run is not None


class EditingService:
    """A page's selected document, backed by shared workflow state."""

    # A span that keeps truncating is halved and retried; past this many
    # attempts hand it to the reviewer rather than shrinking forever.
    MAX_AUTO_SPLITS = 4

    def __init__(self, runtime: EditingRuntime | None = None, *,
                 model_name: str | None = None):
        self.runtime = runtime or EditingRuntime()
        self.model_name = model_name or appstate.state.current_model
        self._state: EditingState | None = None

    @property
    def job(self) -> Job | None:
        return self._state.job if self._state else None

    @property
    def run(self) -> SpanRun | None:
        return self._state.run if self._state else None

    @run.setter
    def run(self, value: SpanRun | None) -> None:
        if self._state is not None:
            self._state.run = value

    @property
    def index(self) -> int | None:
        return self._state.index if self._state else None

    @index.setter
    def index(self, value: int | None) -> None:
        if self._state is not None:
            self._state.index = value

    @property
    def _sent(self) -> tuple[int, list[dict]] | None:
        return self._state.sent if self._state else None

    @_sent.setter
    def _sent(self, value: tuple[int, list[dict]] | None) -> None:
        if self._state is not None:
            self._state.sent = value

    # ── job lifecycle ────────────────────────────────────────────────────────
    def create(self, title: str, source_text: str,
               instructions: Instructions, settings: Settings) -> Outcome:
        raw = store.new_job(title, source_text, instructions.system_prompt,
                            engine.server.model or self.model_name,
                            {**settings.to_dict(), **instructions.to_dict()})
        job = Job.from_dict(raw)
        job.instructions = instructions
        job.settings = settings
        self._adopt(job)
        return self.resume()

    def open(self, job_id: str) -> Outcome:
        live = self.runtime.states.get(job_id)
        if live is not None:
            self._state = live
            return self.resume()
        raw = store.load_job(job_id)
        if raw is None:
            return Outcome(Step.ERROR, "That document could not be loaded.")
        self._adopt(Job.from_dict(raw))
        return self.resume()

    def close(self) -> None:
        """Detach this page without discarding the shared workflow."""
        self._state = None

    def _adopt(self, job: Job) -> None:
        state = EditingState(job=self._prepare(job))
        self._state = self.runtime.remember(state)

    def _prepare(self, job: Job) -> Job:
        """Attach budgets from the loaded server and, on first open, the spans."""
        thinking = (engine.server.reasoning_budget
                    if engine.server.thinking_enabled()
                    and engine.server.reasoning_budget > 0 else 0)
        job.budgets = Budgets.derive(
            job.settings, engine.server.context_limit(),
            spans_mod.est_tokens(job.instructions.system_prompt), thinking)
        if not job.spans:
            job.divide()
            self._save(job)
        return job

    def _save(self, job: Job | None = None) -> None:
        job = job or self.job
        if job is not None:
            job.updated = time.time()
            store.save_job(job.to_dict())

    # ── running ──────────────────────────────────────────────────────────────
    def busy(self) -> bool:
        """True when the single llama-server is already generating something."""
        if self.runtime.busy():
            return True
        return bool(appstate.state.generations)

    def occupied(self, job_id: str) -> bool:
        return self.runtime.occupied(job_id)

    def resume(self) -> Outcome:
        """Pick up at the first span still needing a decision."""
        if self.job is None:
            return Outcome(Step.IDLE)
        # A page attaching to an existing workflow observes and completes this
        # run; it must not start the same span a second time.
        if self.run is not None:
            return Outcome(Step.REVIEW)
        index = self.job.next_undecided()
        if index is None:
            self.index = None
            return Outcome(Step.COMPLETE)
        span = self.job.spans[index]
        if span.status in (Status.PROPOSED, Status.FLAGGED) and span.rewritten:
            self.index = index  # already generated, never decided
            return Outcome(Step.REVIEW)
        return self.begin(index)

    def begin(self, index: int, nudge: str = "") -> Outcome:
        if self.job is None:
            return Outcome(Step.IDLE)
        if not engine.server.running:
            return Outcome(Step.BLOCKED, en.NO_MODEL)
        if self.busy():
            return Outcome(Step.BLOCKED,
                           "The model is busy — wait for the current generation.")
        # Packing sizes spans by character estimate, so a dense one can come out
        # over the reply cap; splitting now is cheaper than a wasted generation.
        if self._oversized(index) and self.job.split(index):
            self._save()
        self.index = index
        messages = self._builder().messages(index, nudge)
        self.run = SpanRun(index, messages, self.job.settings.sampler_params()).start()
        self._sent = (index, messages)
        return Outcome(Step.REVIEW)

    def stop(self) -> None:
        """Halt the current span, and with it any auto-accept chain."""
        if self.run is not None:
            self.run.cancel()

    def retry(self, nudge: str = "") -> Outcome:
        if self.index is None:
            return Outcome(Step.IDLE)
        return self.begin(self.index, nudge)

    def complete_run(self, expected: SpanRun | None = None) -> Outcome:
        """Fold a finished run into its span, then apply the rules.

        May leave a new run in flight (split-retry, or auto-accept advancing),
        so callers should loop while `self.run` is set.
        """
        # Two tabs may observe the same run. The first can complete it and
        # immediately start the next span; the second must not mistake that new
        # run for the one it just finished observing.
        if expected is not None and self.run is not expected:
            return Outcome(Step.REVIEW)
        run, self.run = self.run, None
        if self.job is None or run is None:
            return Outcome(Step.IDLE)
        if run.cancelled:
            # Discard the partial reply and leave the span undecided, so
            # resuming re-runs it rather than half-editing the document.
            return Outcome(Step.CANCELLED,
                           "Stopped. This span is unchanged — press Retry to run "
                           "it again, or come back to it later.")
        span = self._record(run)
        if run.error:
            return Outcome(Step.ERROR, f"Generation error: {run.error}")
        if ("truncated" in span.flags and span.attempts <= self.MAX_AUTO_SPLITS
                and self.job.split(run.index)):
            self._save()
            self.begin(run.index)
            return Outcome(Step.SPLIT_RETRY,
                           "Span was too long for one reply — split and retrying.")
        if not span.flags and self.job.settings.auto_accept_clean:
            self.job.decide(run.index, Status.ACCEPTED)
            self._save()
            return self.advance()
        return Outcome(Step.REVIEW)

    def advance(self) -> Outcome:
        if self.job is None:
            return Outcome(Step.IDLE)
        start = self.index if self.index is not None else -1
        index = self.job.next_undecided(start)
        if index is None:
            self.index = None
            return Outcome(Step.COMPLETE)
        self.begin(index)
        return Outcome(Step.ADVANCED)

    # ── decisions ────────────────────────────────────────────────────────────
    def _undecidable(self) -> Outcome | None:
        """Why a decision can't be made right now, or None if it can."""
        if self.job is None or self.index is None:
            return Outcome(Step.IDLE)
        if self.run is not None:
            return Outcome(Step.BLOCKED, en.SPAN_BUSY)
        return None

    def accept(self, text: str | None = None) -> Outcome:
        blocked = self._undecidable()
        if blocked:
            return blocked
        span = self.job.spans[self.index]
        if text is None and not span.rewritten.strip():
            # Reachable after a stopped or failed run: accepting would put
            # nothing in place of the passage and quietly delete it.
            return Outcome(Step.BLOCKED,
                           "There's no rewrite to accept — press Retry, or Keep "
                           "original to leave this passage alone.")
        if text is not None and not text.strip():
            return Outcome(Step.BLOCKED, "The rewrite can't be empty.")
        self.job.decide(self.index, Status.ACCEPTED, text)
        self._save()
        return self.advance()

    def keep_original(self) -> Outcome:
        blocked = self._undecidable()
        if blocked:
            return blocked
        self.job.decide(self.index, Status.ORIGINAL)
        self._save()
        return self.advance()

    # ── views for the page ───────────────────────────────────────────────────
    @property
    def span(self) -> Span | None:
        if self.job is None or self.index is None:
            return None
        return self.job.spans[self.index]

    @property
    def running(self) -> bool:
        return self.run is not None

    def live_text(self) -> str:
        """What to show in the rewrite pane right now."""
        if self.run is not None:
            return self.run.text
        span = self.span
        return span.rewritten if span else ""

    def export_text(self) -> str:
        return self.job.assemble() if self.job else ""

    def prompt_messages(self) -> tuple[list[dict], bool]:
        """Messages for the open span, and whether they are what was sent."""
        if self.job is None or self.index is None:
            return [], False
        if self._sent and self._sent[0] == self.index:
            return self._sent[1], True
        return self._builder().messages(self.index), False

    def check(self, index: int, text: str, finish_reason: str | None) -> list[str]:
        """Validate a rewrite in the context of its neighbours."""
        before, after = self._builder().context(index)
        return self._validator().check(
            self.job.spans[index].original, text, finish_reason,
            before=before, after=after)

    # ── internals ────────────────────────────────────────────────────────────
    def _builder(self) -> PromptBuilder:
        return PromptBuilder(self.job, engine.server.supports_system_role())

    def _validator(self) -> Validator:
        return Validator(allow_deletions=self.job.settings.allow_deletions)

    def _oversized(self, index: int) -> bool:
        """Exact token check just before send; packing only estimates."""
        count, exact = engine.server.count_tokens(self.job.spans[index].original)
        return exact and count > self.job.budgets.span_target

    def _record(self, run: SpanRun) -> Span:
        span = self.job.spans[run.index]
        span.attempts += 1
        span.finish_reason = run.finish_reason
        if run.error:
            span.status = Status.FLAGGED
            span.flags = ["error"]
            span.rewritten = ""
        else:
            span.rewritten = reattach_edges(span.original, run.text)
            span.flags = self.check(run.index, span.rewritten, run.finish_reason)
            span.status = Status.FLAGGED if span.flags else Status.PROPOSED
        self._save()
        self._log(run, span)
        return span

    def _log(self, run: SpanRun, span: Span) -> None:
        """Append one attempt to $THAUM_LOG_DIR/editing.jsonl. Never raises."""
        directory = log_dir()
        if directory is None:
            return
        try:
            prompt = "".join(m["content"] for m in run.messages)
            elapsed = run.elapsed
            validator = self._validator()
            record = {
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "job": self.job.id,
                "span": run.index,
                "attempt": span.attempts,
                "seconds": round(elapsed, 1),
                "prompt_msgs": len(run.messages),
                "prompt_tok_est": spans_mod.est_tokens(prompt),
                "overlap_tok": self.job.budgets.overlap if self.job.budgets else None,
                "span_chars": len(span.original),
                "out_chars": len(span.rewritten),
                "out_tok_per_s": (round(spans_mod.est_tokens(span.rewritten) / elapsed, 1)
                                  if elapsed else None),
                "finish": span.finish_reason,
                "flags": span.flags,
                "drawn_from_source": round(
                    validator.drawn_from_source(span.original, span.rewritten), 3),
                "reaches_end": validator.reaches_end(span.original, span.rewritten),
                "error": run.error,
            }
            with (directory / "editing.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except (OSError, KeyError, TypeError, ValueError, AttributeError):
            pass


editing_runtime = EditingRuntime()
