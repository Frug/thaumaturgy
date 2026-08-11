"""Folding the oldest turns of a chat into one recap, to free context.

The transcript on disk is never edited. Compaction only appends a Summary
record naming how many leading messages it stands for; `prompt.build` swaps
them for its text when assembling a request, and the user's view stays whole.

Re-compaction is incremental: the input is the previous recap plus the turns
that have arrived since, so a long chat never re-reads its whole history.
"""

import time
from dataclasses import dataclass

from thaumaturgy import appstate, engine, store
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
# No model reports how long a summary it will write, so the budget is a guess
# the user tunes. Capped as a share of the window because the recap is sent
# with every later turn: past this it starts crowding out the recent messages
# it was meant to make room for.
MAX_RECAP_SHARE = 0.15
MIN_RECAP_TOKENS = 256
MIN_KEEP = 4                # messages left verbatim, however big they are
MIN_FOLD = 4                # fewer than this isn't worth a round trip
WORDS_PER_TOKEN = 0.75      # recap budget is stated to the model in words
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
    recap_tokens: int

    @property
    def compacted(self) -> bool:
        return self.covered > 0


def window() -> int | None:
    """The context window replies have to fit in, before or after a load."""
    if engine.server.running:
        return engine.server.context_limit()
    model = engine.server.model or appstate.state.current_model
    return engine.trained_ctx(model) if model else None


def recap_budget(total: int) -> int:
    """How long the recap may run: the parameter set's value, capped to the window."""
    params = appstate.state.current_params or {}
    try:
        wanted = int(params.get("recap_tokens", RECAP_TOKENS_DEFAULT))
    except (TypeError, ValueError):
        wanted = RECAP_TOKENS_DEFAULT
    return max(MIN_RECAP_TOKENS, min(wanted, int(total * MAX_RECAP_SHARE)))


def reserve() -> int:
    """Tokens to keep free for the reply itself."""
    params = appstate.state.current_params or {}
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
                      covered=0, recap_tokens=0)
    # Only a compacted chat pays for the second count.
    full, _ = _count(chat, scenario, draft, supports_system_role, compacted=False)
    return Report(used=used, total=window(), exact=exact, full=full,
                  covered=summary.covers, recap_tokens=summary.tokens)


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
         supports_system_role: bool = True) -> Plan | None:
    """What compaction is needed before the next reply, or None if it isn't.

    A returned Plan may still be impossible (`possible` False) when the recent
    turns alone fill the window; the caller has to say so rather than compact.
    """
    if chat is None or not chat.messages:
        return None
    total = window()
    if not total:
        return None
    room = reserve()
    used, _ = _count(chat, scenario, draft, supports_system_role, compacted=True)
    if used + room <= total * TRIGGER_RATIO:
        return None

    budget = recap_budget(total)
    overhead = engine.estimate_tokens(scenario.context) if scenario else 0
    keep = int(total * TARGET_RATIO) - room - budget - overhead
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


def _fill(template: str, values: dict) -> str:
    for key, value in values.items():
        token = "{" + key + "}"
        if token in template:
            template = template.replace(token, value)
        elif value.strip():
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


def run(chat: Chat, scenario: Scenario | None, target: Plan,
        supports_system_role: bool = True) -> Summary:
    """Write the recap for `target`. Blocking: it is a generation of its own."""
    doc = store.load_compaction_prompt()
    previous = chat.active_summary()
    carried = ""
    if previous is not None and previous.text.strip():
        carried = (doc["carry"] + "\n" + previous.text.strip())

    room = target.total - target.budget - REQUEST_OVERHEAD \
        - engine.estimate_tokens(doc["system"] + doc["instruction"] + carried)
    transcript = _transcript(chat.messages[target.start:target.covers], max(room, 512))
    if not transcript.strip():
        raise RuntimeError("Those messages have no text to summarize.")

    ask = _fill(doc["instruction"], {
        "max_words": str(int(target.budget * WORDS_PER_TOKEN)),
        "scenario": scenario.name if scenario else "",
        "recap": carried,
        "transcript": transcript,
    })
    system = doc["system"].strip()
    if supports_system_role:
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": ask}]
    else:
        messages = [{"role": "user", "content": f"{system}\n\n{ask}".strip()}]

    text = _generate(messages, target.budget).strip()
    if not text:
        raise RuntimeError("The model returned an empty recap.")
    tokens, _ = engine.server.count_tokens(text)
    return Summary(text=text, covers=target.covers,
                   fingerprint=fingerprint(chat.messages[:target.covers]),
                   tokens=tokens,
                   model=engine.server.model or appstate.state.current_model,
                   created=time.time())
