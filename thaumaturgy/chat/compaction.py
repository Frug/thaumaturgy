"""Folding the oldest turns of a chat into one recap, to free context.

The transcript on disk is never edited. Compaction only appends a Summary
record naming how many leading messages it stands for; `prompt.build` swaps
them for its text when assembling a request, and the user's view stays whole.

Re-compaction is incremental: only the turns since the last recap are folded,
so a long chat never re-reads its whole history. Several passes keep that recap
as it stands and write beside it; one pass condenses it again along with the
new turns. A target starting at message zero rebuilds, keeping nothing.
"""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

from thaumaturgy import appstate, engine, prompting, store
from thaumaturgy.chat import prompt, reply
from thaumaturgy.chat.models import Chat, Message, Role, Scenario, Summary, fingerprint

# Of the window, counting the reply we are about to make room for. Below this
# nothing happens; the cost of compacting is a full prompt reprocess, since
# rewriting the prefix drops llama.cpp's prompt cache.
TRIGGER_RATIO = 0.85
# What the prompt should shrink to. The gap from TRIGGER_RATIO is deliberate:
# compact rarely and deeply rather than trimming a turn per message.
TARGET_RATIO = 0.5
RECAP_TOKENS_DEFAULT = 4000  # when the preset predates the setting
# The recap is resent with every later turn, so past this share of the window
# it crowds out the recent messages it was meant to make room for.
MAX_RECAP_SHARE = 0.15
MIN_RECAP_TOKENS = 256
MIN_KEEP = 4                # messages left verbatim, however big they are
MIN_FOLD = 4                # fewer than this isn't worth a round trip
WORDS_PER_TOKEN = 0.75      # recap budget is stated to the model in words
MIN_WORDS_SHARE = 0.6       # of the budget, asked for as a floor
# Transcript one pass is asked to condense; a longer fold is split across
# several passes, one generation each.
PASS_INPUT_TOKENS = 10_000
MAX_PASSES = 8              # each is a generation, so this bounds the wait
MIN_PASS_TOKENS = 300       # a share too small to say anything useful with
REQUEST_OVERHEAD = 256      # template scaffolding around the summarizer's own prompt


@dataclass(frozen=True)
class Plan:
    """What one compaction would fold in, and how much room it would win."""

    start: int      # first message no recap covers yet
    covers: int     # messages the new recap would stand for
    used: int       # prompt tokens as things stand
    total: int      # the context window
    budget: int     # tokens allowed for the recap

    @property
    def folded(self) -> int:
        return self.covers - self.start

    @property
    def possible(self) -> bool:
        return self.folded >= MIN_FOLD


@dataclass(frozen=True)
class Report:
    """Context accounting for the meter: what the model gets, and what's behind it."""

    used: int
    total: int | None
    exact: bool
    full: int           # what the same chat would cost with no recap
    covered: int        # messages the recap stands in for
    messages: int       # messages in the chat, folded or not
    recap_tokens: int

    @property
    def compacted(self) -> bool:
        return self.covered > 0

    @property
    def verbatim(self) -> int:
        return self.messages - self.covered


def window() -> int | None:
    """The context window replies have to fit in, before or after a load."""
    if engine.server.running:
        return engine.server.context_limit()
    model = engine.server.model or appstate.state.current_model
    return engine.trained_ctx(model) if model else None


def recap_budget(total: int, params: dict | None = None) -> int:
    """How long the recap may run: the parameter set's value, capped to the window."""
    params = appstate.state.current_params if params is None else params
    try:
        wanted = int(params.get("recap_tokens", RECAP_TOKENS_DEFAULT))
    except (TypeError, ValueError):
        wanted = RECAP_TOKENS_DEFAULT
    return max(MIN_RECAP_TOKENS, min(wanted, int(total * MAX_RECAP_SHARE)))


def reserve(params: dict | None = None) -> int:
    """Tokens to keep free for the reply itself."""
    params = appstate.state.current_params if params is None else params
    try:
        room = int(params.get("max_new_tokens", 512))
    except (TypeError, ValueError):
        room = 512
    budget = engine.server.reasoning_budget
    if engine.server.thinking_enabled() and budget > 0:
        room += budget
    return room


def _count(chat: Chat, scenario: Scenario | None, draft: str,
           supports_system_role: bool, *, compacted: bool) -> tuple[int, bool]:
    return engine.server.count_chat_tokens(
        prompt.build(chat, scenario, draft=draft, compacted=compacted,
                     supports_system_role=supports_system_role))


def report(chat: Chat | None, scenario: Scenario | None, draft: str = "",
           supports_system_role: bool = True) -> Report:
    chat = chat or Chat(id="")
    summary = chat.active_summary()
    used, exact = _count(chat, scenario, draft, supports_system_role, compacted=True)
    if summary is None:
        return Report(used=used, total=window(), exact=exact, full=used,
                      covered=0, messages=len(chat.messages), recap_tokens=0)
    # Only a compacted chat pays for the second count.
    full, _ = _count(chat, scenario, draft, supports_system_role, compacted=False)
    return Report(used=used, total=window(), exact=exact, full=full,
                  covered=summary.covers, messages=len(chat.messages),
                  recap_tokens=summary.tokens)


def _split_point(messages: list[Message], keep_tokens: int, start: int) -> int:
    """Where the verbatim tail should begin, walking back from the newest turn."""
    index = len(messages)
    kept = 0
    while index > start:
        size = engine.estimate_tokens(messages[index - 1].text or "")
        if kept + size > keep_tokens and len(messages) - index >= MIN_KEEP:
            break
        kept += size
        index -= 1
    # Open the tail on a user turn, so the recap doesn't cut an exchange in half
    # and the history the model sees reads as a reply to something.
    while index < len(messages) - 1 and messages[index].role is not Role.USER:
        index += 1
    return index


def plan(chat: Chat | None, scenario: Scenario | None, *, draft: str = "",
         supports_system_role: bool = True, force: bool = False,
         params: dict | None = None) -> Plan | None:
    """What compaction is needed before the next reply, or None if it isn't.

    With `force`, plan one regardless of how full the window is: the user asking
    for it outright is reason enough. A returned Plan may still be impossible
    (`possible` False) when the recent turns alone fill the window; the caller
    has to say so rather than compact.
    """
    if chat is None or not chat.messages:
        return None
    total = window()
    if not total:
        return None
    room = reserve(params)
    used, _ = _count(chat, scenario, draft, supports_system_role, compacted=True)
    if not force and used + room <= total * TRIGGER_RATIO:
        return None

    budget = recap_budget(total, params)
    overhead = engine.estimate_tokens(scenario.context) if scenario else 0
    keep = int(total * TARGET_RATIO) - room - budget - overhead
    if force:
        # The target is a share of the window, so a chat that still fits would
        # plan to keep all of itself. Fold the older half of what is there.
        keep = min(keep, used // 2)
    summary = chat.active_summary()
    start = summary.covers if summary is not None else 0
    covers = _split_point(chat.messages, max(keep, 0), start)
    return Plan(start=start, covers=covers, used=used, total=total, budget=budget)


def _speaker(m: Message) -> str:
    return m.name or ("You" if m.role is Role.USER else "Assistant")


def _transcript(messages: list[Message], limit: int) -> str:
    """The turns to fold in, as speaker-labelled lines within a token budget.

    Trimmed from the front when it doesn't fit: the head is the part the
    previous recap already covers most closely.
    """
    lines, spent, dropped = [], 0, 0
    for taken, m in enumerate(reversed(messages)):
        text = (m.text or "").strip()
        if not text:
            continue
        line = f"{_speaker(m)}: {text}"
        size = engine.estimate_tokens(line)
        if spent + size > limit and lines:
            dropped = len(messages) - taken
            break
        lines.append(line)
        spent += size
    lines.reverse()
    if dropped:
        lines.insert(0, f"[{dropped} earlier turns omitted]")
    return "\n\n".join(lines)


def _fill(template: str, values: dict, content: tuple = ("recap", "transcript")) -> str:
    """Substitute placeholders, appending only the ones that carry the material.

    An edited template that drops {transcript} would otherwise leave the model
    nothing to work from; one that drops {turns} just doesn't want the number.
    """
    for key, value in values.items():
        token = "{" + key + "}"
        if token in template:
            template = template.replace(token, value)
        elif key in content and value.strip():
            template = f"{template}\n\n{value}"
    return template


def _generate(messages: list[dict], budget: int) -> str:
    """Run the summarizer to completion and return its text."""
    params = {"temperature": 0.3, "top_p": 0.9, "top_k": 40, "min_p": 0.05,
              "repetition_penalty": 1.05, "max_new_tokens": budget}
    text, reasoning = "", ""
    for event in engine.server.stream_chat(messages, params):
        delta = event.get("text", "")
        if not delta:
            continue
        if event.get("type") == "reasoning":
            reasoning += delta
        else:
            text += delta
    # Templates llama.cpp can't parse leave their channel markers in the text.
    return reply.interpret(text, reasoning)[0].strip()


def _split_into_passes(messages: list[Message], limit: int,
                       cap: int = MAX_PASSES) -> list[list[Message]]:
    """Break the fold into spans of at most `limit` tokens each, `cap` of them.

    Spans are sized by what goes in, not by what should come out: each gets a
    generation of its own, so their number sets how many times the model is
    asked and how the recap budget is divided.
    """
    passes, current, spent = [], [], 0
    for m in messages:
        size = engine.estimate_tokens(m.text or "")
        if current and spent + size > limit:
            passes.append(current)
            current, spent = [], 0
        current.append(m)
        spent += size
    if current:
        passes.append(current)
    if len(passes) <= cap:
        return passes
    # Too many for the time they would take, or for the budget to divide
    # usefully: regroup into `cap` even spans.
    per = math.ceil(len(messages) / cap)
    return [messages[i:i + per] for i in range(0, len(messages), per)]


def _pass_budgets(passes: list[list[Message]], budget: int) -> list[int]:
    """Divide the recap budget between passes, weighted by how much each holds.

    Spans come out uneven, the last one holding whatever was left, so an even
    split would give a handful of turns the same room as a full span. The total
    stays within the budget.
    """
    weights = [max(engine.estimate_tokens("\n".join(m.text or "" for m in span)), 1)
               for span in passes]
    total = sum(weights)
    shares = [max(MIN_PASS_TOKENS, int(budget * w / total)) for w in weights]
    # Raising a small span to its floor is paid for by the passes with room to
    # spare, in proportion, so the total still fits.
    over = sum(shares) - budget
    spare = [s - MIN_PASS_TOKENS for s in shares]
    if over > 0 and sum(spare) > 0:
        givable = sum(spare)
        shares = [s - math.ceil(over * room / givable)
                  for s, room in zip(shares, spare)]
    return shares


def run(chat: Chat, scenario: Scenario | None, target: Plan,
        supports_system_role: bool = True,
        on_progress: Callable[[int, int], None] | None = None) -> Summary:
    """Write the recap for `target`. Blocking: these are generations of its own.

    Long folds are summarized in several passes, one span at a time, and the
    parts are kept in order. The budget is shared between them.
    """
    doc = store.load_compaction_prompt()
    one_pass = store.compaction_strategy() == "single"
    # A target starting at 0 is a rebuild: the recap the chat has is ignored.
    previous = chat.active_summary() if target.start > 0 else None
    start, budget = target.start, target.budget
    kept, carried = "", ""
    if previous is not None and previous.text.strip():
        # Recaps written before the count was recorded fall back to an estimate.
        held = previous.tokens or engine.estimate_tokens(previous.text)
        if one_pass:
            # The one generation has to hold the whole recap, earlier part
            # included, so that part is condensed again.
            carried = (doc["carry"] + "\n" + previous.text.strip())
        elif budget - held >= MIN_PASS_TOKENS:
            # A pass only sees its own span, so the earlier recap can stand as
            # it is. Re-condensing it thins it out a little every time.
            kept = previous.text.strip()
            budget -= held
        else:
            # No room to write beside it: rebuild rather than overrun the budget.
            start = 0

    folded = chat.messages[start:target.covers]
    # One generation covers the whole fold, or one per span of it. A small
    # budget takes fewer spans, since shares below MIN_PASS_TOKENS say nothing.
    passes = ([folded] if one_pass
              else _split_into_passes(
                  folded, PASS_INPUT_TOKENS,
                  cap=max(1, min(MAX_PASSES, budget // MIN_PASS_TOKENS))))
    budgets = _pass_budgets(passes, budget)
    system = doc["system"].strip()

    multi = len(passes) > 1 or bool(kept)
    parts, first_turn = [], start + 1
    if kept:
        # A recap written in one pass has no heading of its own.
        parts.append(kept if kept.lstrip().startswith("## Turns")
                     else f"## Turns 1-{start}\n\n{kept}")
    for index, span in enumerate(passes):
        if on_progress is not None:
            on_progress(index + 1, len(passes))
        share = budgets[index]
        room = target.total - share - REQUEST_OVERHEAD \
            - engine.estimate_tokens(system + doc["instruction"] + carried)
        transcript = _transcript(span, max(room, 512))
        last_turn = first_turn + len(span) - 1
        # An empty span is skipped, unless it carries the earlier recap.
        if not transcript.strip() and not carried:
            first_turn = last_turn + 1
            continue
        max_words = int(share * WORDS_PER_TOKEN)
        ask = _fill(doc["instruction"], {
            "max_words": str(max_words),
            # A floor as well as a ceiling: asked only for a maximum, models
            # treat the task as done long before the budget is near spent.
            "min_words": str(max(120, int(max_words * MIN_WORDS_SHARE))),
            "turns": str(len(span)),
            "scenario": scenario.name if scenario else "",
            "recap": carried,
            "transcript": transcript,
        })
        messages = prompting.with_system([{"role": "user", "content": ask}],
                                         system, supports_system_role)
        part = _generate(messages, share).strip()
        if part:
            parts.append(f"## Turns {first_turn}-{last_turn}\n\n{part}"
                         if multi else part)
        first_turn = last_turn + 1

    text = "\n\n".join(parts).strip()
    if not text:
        raise RuntimeError("The model returned an empty recap.")
    tokens, _ = engine.server.count_tokens(text)
    return Summary(text=text, covers=target.covers,
                   fingerprint=fingerprint(chat.messages[:target.covers]),
                   tokens=tokens,
                   model=engine.server.model or appstate.state.current_model,
                   created=time.time())
