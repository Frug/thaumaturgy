"""Span-by-span document editing.

A model capped at a few hundred output tokens cannot rewrite a long document in
one reply, so the loop is inverted: a cursor walks the document and each
generation rewrites one bounded span, with the surrounding text supplied as
read-only context. Accepted rewrites are spliced back in and become context for
the spans below them.

How that prompt is laid out matters more than anything else here. Surrounding
text goes in an earlier, already-answered turn and the passage arrives alone in
the final one, because a model handed one blob of context with the passage
fenced off inside it starts reproducing at the top of the blob rather than at
the passage. Keeping the window narrow matters for the same reason: the more
neighbouring prose is in front of the model, the likelier it wanders into it,
and `overlap` of 0 — passage and nothing else — is the most faithful setting
there is. What context buys in return is consistent terminology across the
joins, so it is worth a little, and only a little.

No NiceGUI here: the page is presentation only.
"""

import math
import re
import threading
from difflib import SequenceMatcher

from thaumaturgy import appstate, engine, store

# Rough bytes-per-token for packing. Only needs to be close: spans are verified
# exactly before they are sent, and an over-long one is split.
CHARS_PER_TOKEN = 4

# Floor on splitting, so repeated truncation can't shave a span down to nothing.
MIN_SPAN_CHARS = 40

# Headroom for the chat template, the restated span, and the instruction text.
PROMPT_OVERHEAD = 256
SAFETY_RESERVE = 256

DEFAULT_SETTINGS = {
    "max_new_tokens": 700,
    "temperature": 0.2,
    "top_p": 0.95,
    "top_k": 40,
    "min_p": 0.05,
    "repetition_penalty": 1.05,
    "response_buffer": 150,   # how far under the reply cap to size a span
    # Off by default. Neighbouring prose is the single biggest cause of a model
    # wandering out of the passage it was given; raise it only if your model
    # holds up, and watch the first spans when you do.
    "overlap_pct": 0.0,
    "auto_accept_clean": False,
    "allow_deletions": False,  # whether the instructions may remove text
}

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful copy editor. Fix grammar, spelling, and punctuation. "
    "Preserve the author's voice, meaning, and paragraph structure. Do not "
    "add, remove, or reorder content."
)

PENDING, PROPOSED, ACCEPTED, ORIGINAL, FLAGGED = (
    "pending", "proposed", "accepted", "original", "flagged")


def normalize_settings(vals) -> dict:
    """Coerce anything into a full settings dict; the job files are editable."""
    src = vals if isinstance(vals, dict) else {}
    out = dict(DEFAULT_SETTINGS)
    for key, cast in (("max_new_tokens", int), ("response_buffer", int),
                      ("top_k", int), ("temperature", float), ("top_p", float),
                      ("min_p", float), ("repetition_penalty", float),
                      ("overlap_pct", float)):
        try:
            out[key] = cast(src.get(key, out[key]))
        except (TypeError, ValueError):
            pass
    for key in ("auto_accept_clean", "allow_deletions"):
        out[key] = bool(src.get(key, out[key]))
    out["overlap_pct"] = min(0.45, max(0.0, out["overlap_pct"]))
    out["response_buffer"] = max(0, min(out["response_buffer"],
                                        out["max_new_tokens"] - 64))
    return out


def sampler_params(settings: dict) -> dict:
    """The subset of settings that engine.stream_chat consumes."""
    return {k: settings[k] for k in
            ("max_new_tokens", "temperature", "top_p", "top_k", "min_p",
             "repetition_penalty")}


def est_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


# ── Splitting ───────────────────────────────────────────────────────────────
# Spans are (start, end) offsets that concatenate back to the source exactly, so
# reassembly can never lose or duplicate text. Boundaries land after sentence
# punctuation, at paragraph breaks, and before markdown blocks — but never at a
# bare newline, which in hard-wrapped prose is a soft wrap rather than a sentence
# end. A span that opens mid-sentence makes the model complete the fragment from
# the context above it, duplicating those words at the join.

_BOUNDARY_RE = re.compile(
    r"[.!?][\"'’”)\]]*\s+"                    # sentence end, with the gap after it
    r"|\n[ \t]*\n\s*"                          # blank line: a paragraph break
    r"|\n(?=[ \t]*(?:[-*+>#]|\d+[.)])[ \t])")  # break before a markdown block


def split_atoms(text: str) -> list[tuple[int, int]]:
    """Cut `text` at sentence/line boundaries into contiguous atoms."""
    out, pos = [], 0
    for m in _BOUNDARY_RE.finditer(text):
        if m.end() > pos:
            out.append((pos, m.end()))
            pos = m.end()
    if pos < len(text):
        out.append((pos, len(text)))
    return out


def _hard_split(text: str, start: int, end: int, budget: int) -> list[tuple[int, int]]:
    """Break an atom that alone exceeds the budget, preferring whitespace."""
    out, pos = [], start
    while end - pos > budget:
        cut = text.rfind(" ", pos + budget // 2, pos + budget)
        cut = cut + 1 if cut > pos else pos + budget
        out.append((pos, cut))
        pos = cut
    if pos < end:
        out.append((pos, end))
    return out


def pack_spans(text: str, budget_chars: int) -> list[tuple[int, int]]:
    """Greedily group atoms into spans of at most `budget_chars`."""
    atoms: list[tuple[int, int]] = []
    for a_start, a_end in split_atoms(text):
        if a_end - a_start > budget_chars:
            atoms.extend(_hard_split(text, a_start, a_end, budget_chars))
        else:
            atoms.append((a_start, a_end))
    out: list[tuple[int, int]] = []
    start = end = None
    for a_start, a_end in atoms:
        if start is None:
            start, end = a_start, a_end
        elif a_end - start > budget_chars:
            out.append((start, end))
            start, end = a_start, a_end
        else:
            end = a_end
    if start is not None:
        out.append((start, end))
    # Fold a runt tail back into its neighbour. Packing leaves one whenever the
    # text divides unevenly — most visibly when split_span halves an over-long
    # span. A lone sentence costs a whole generation and gives the model so
    # little to work from that it treats the fragment as a writing prompt and
    # returns fresh prose instead of a correction. Skipped when merging would
    # overshoot the budget enough to risk truncating the result.
    if len(out) > 1:
        runt = out[-1][1] - out[-1][0]
        merged = out[-1][1] - out[-2][0]
        if runt < max(MIN_SPAN_CHARS, budget_chars // 4) and merged <= budget_chars * 1.15:
            tail = out.pop()
            out[-1] = (out[-1][0], tail[1])
    return out


def make_spans(text: str, span_target_tokens: int) -> list[dict]:
    budget_chars = max(200, span_target_tokens * CHARS_PER_TOKEN)
    return [new_span(text, s, e) for s, e in pack_spans(text, budget_chars)]


def new_span(text: str, start: int, end: int) -> dict:
    return {
        "start": start,
        "end": end,
        "original": text[start:end],
        "rewritten": "",
        "status": PENDING,
        "finish_reason": None,
        "flags": [],
        "attempts": 0,
    }


def split_span(job: dict, index: int) -> bool:
    """Replace one span with two, for a span that proved too big to rewrite.

    Returns False once the span is too small to divide usefully, so a model that
    rambles no matter what can't drive the split down to single characters.
    """
    span = job["spans"][index]
    text = job["source_text"]
    if span["end"] - span["start"] < 2 * MIN_SPAN_CHARS:
        return False
    half = max(MIN_SPAN_CHARS, (span["end"] - span["start"]) // 2)
    parts = pack_spans(span["original"], half)
    if len(parts) < 2:
        return False
    base = span["start"]
    job["spans"][index:index + 1] = [
        new_span(text, base + s, base + e) for s, e in parts]
    return True


# ── Budgets and chunking ────────────────────────────────────────────────────

def budgets(settings: dict, context_limit: int | None,
            system_tokens: int = 0) -> dict:
    """Token allowances for one job, derived from the loaded server."""
    max_new = settings["max_new_tokens"]
    span_target = max(64, max_new - settings["response_buffer"])
    reserve = max_new + system_tokens + SAFETY_RESERVE
    if engine.server.thinking_enabled() and engine.server.reasoning_budget > 0:
        reserve += engine.server.reasoning_budget
    usable = max(1024, (context_limit or 8192) - reserve)
    # The span occupies the prompt twice: marked in place, and restated at the
    # tail for the model to copy from.
    available = max(512, usable - 2 * span_target - PROMPT_OVERHEAD)
    overlap = int(available * settings["overlap_pct"])
    return {"span_target": span_target, "available": available,
            "overlap": overlap}


def current_text(span: dict) -> str:
    """What this span contributes to the document as it stands."""
    return span["rewritten"] if span["status"] == ACCEPTED else span["original"]


def assemble(job: dict) -> str:
    """The document as edited so far — undecided spans fall back to original."""
    return "".join(current_text(s) for s in job["spans"])


def _window(job: dict, index: int) -> tuple[int, int]:
    """First and last span index (inclusive) of the prompt body around `index`.

    Centred on the span itself: `overlap` tokens of corrected text behind it and
    `overlap` tokens of original text ahead, and nothing more. Reaching wider
    buries the passage in prose that reads just like it, and a model asked to
    reproduce the marked part instead wanders off into the rest.
    """
    spans = job["spans"]
    overlap = job["budgets"]["overlap"]
    start, used = index, 0
    while start > 0 and used + est_tokens(current_text(spans[start - 1])) <= overlap:
        start -= 1
        used += est_tokens(current_text(spans[start]))
    end, used = index, 0
    while end < len(spans) - 1 and used + est_tokens(spans[end + 1]["original"]) <= overlap:
        end += 1
        used += est_tokens(spans[end]["original"])
    return start, end


def build_messages(job: dict, index: int, nudge: str = "") -> list[dict]:
    """Assemble the chat messages that ask for one span's rewrite.

    Context goes in an earlier, answered turn; the passage arrives alone in the
    final one. Splicing the passage into the middle of a single context blob —
    even fenced off with markers — reliably makes the model start reproducing at
    the top of the blob instead of at the passage. Measured against that layout,
    this one took a failing span from 0.18 of its output tracing back to the
    source to 0.99, and a span with no context at all reaches 1.00.
    """
    spans = job["spans"]
    start, end = _window(job, index)
    before = "".join(current_text(s) for s in spans[start:index])
    after = "".join(s["original"] for s in spans[index + 1:end + 1])
    target = spans[index]["original"]

    instruction = (
        "Output only the corrected form of the passage below. Reproduce it in "
        "full, with no commentary, no labels, and none of the surrounding text."
    )
    if nudge.strip():
        instruction += f"\n\nAdditional instruction: {nudge.strip()}"

    turns = []
    reference = []
    if before.strip():
        reference.append(f"--- text before the passage ---\n{before}")
    if after.strip():
        reference.append(f"--- text after the passage ---\n{after}")
    if reference:
        joined = "\n\n".join(reference)
        turns += [
            {"role": "user", "content":
             "I will give you a passage to edit. First, for reference only, "
             f"here is the text that surrounds it in the document.\n\n{joined}"},
            {"role": "assistant", "content":
             "Understood. I have read the surrounding text for reference only "
             "and will not reproduce any of it. Send me the passage to edit."},
        ]
    turns.append({"role": "user", "content": f"{instruction}\n\n{target}"})

    system = (job.get("system_prompt") or "").strip()
    if not system:
        return turns
    if engine.server.supports_system_role():
        return [{"role": "system", "content": system}, *turns]
    turns[0]["content"] = f"{system}\n\n{turns[0]['content']}"
    return turns


# ── Output guards ───────────────────────────────────────────────────────────

MAX_RATIO = 2.0
MIN_RATIO = 0.5
# Deleting text is a legitimate instruction ("strip the scraped page furniture"),
# and it shrinks a span on purpose. When the job allows it, only a near-total
# gutting is worth a second look.
MIN_RATIO_DELETING = 0.1
_RATIO_FLOOR = 40
_BLEED_WINDOW = 60
# Shortest run of shared text worth treating as a real alignment, not noise.
_ALIGN_BLOCK = 20

# How far through the original the rewrite's closing words must fall. Catches a
# model that stops partway and drops the rest of the span — invisible to every
# length check once the job allows deletions, since both just come back short.
MIN_END_COVERAGE = 0.8

# Least share of the *output* that must trace back to the original. Measured:
# a pure deletion scores 1.00 and a light copy edit 0.98, while prose the model
# invented lands near 0.15 — so this separates invention from removal, which a
# symmetric similarity cannot. Length carries the rest: a summary reuses enough
# wording to pass here (0.79) but collapses the ratio, so MIN_RATIO catches it.
MIN_DRAWN_FROM_SOURCE = 0.6


def _flat(text: str) -> str:
    return " ".join(text.split())


def reaches_end(original: str, text: str) -> bool:
    """Whether the rewrite carries through to the end of the span.

    A model that gives up halfway returns exactly what one that deleted a lot
    returns — both simply come back short, and once a job allows deletions no
    length check can separate them. But an honest edit still finishes where the
    span finishes, so find the output's closing words in the original and see
    how far through they fall.
    """
    a, b = _flat(original), _flat(text)
    if len(a) < _BLEED_WINDOW or len(b) < _BLEED_WINDOW:
        return True
    # Aligned rather than searched for: hunting the closing words directly finds
    # their *last* occurrence, which in repetitive prose sits at the end however
    # early the model actually stopped. An alignment advances monotonically, so
    # its final block lands where the output genuinely ran out.
    blocks = [bl for bl in SequenceMatcher(None, a, b).get_matching_blocks()
              if bl.size >= _ALIGN_BLOCK]
    if not blocks:
        return True  # nothing of the author's survives; that's invention's to flag
    last = blocks[-1]
    return (last.a + last.size) / len(a) >= MIN_END_COVERAGE


def drawn_from_source(original: str, text: str) -> float:
    """Share of `text` that traces back to `original`, 0..1.

    Deliberately asymmetric. A symmetric similarity punishes deletion, which is
    a legitimate instruction — telling the model to strip scraped page furniture
    should not look like a defect. Removing text leaves this at 1.0, because
    every character that remains still came from the source; inventing prose
    drives it down, because the new words match nothing.
    """
    a, b = _flat(original), _flat(text)
    if not b:
        return 1.0
    matched = sum(block.size for block in
                  SequenceMatcher(None, a, b).get_matching_blocks())
    return matched / len(b)


def reattach_edges(original: str, text: str) -> str:
    """Restore the span's own leading and trailing whitespace.

    Models drop or reflow boundary whitespace no matter how the prompt asks, and
    a newline lost at a span join silently welds two paragraphs into one. Those
    edges carry the document's structure rather than any of its content, so take
    them from the source instead of trusting the reply.
    """
    lead = original[:len(original) - len(original.lstrip())]
    trail = original[len(original.rstrip()):]
    return lead + text.strip() + trail


def _bleeds(edge: str, text: str, at_start: bool) -> bool:
    """True when the rewrite runs on into the text beyond the span's markers.

    A bleed means the output opens with a *suffix* of the preceding context (or
    closes with a *prefix* of the following one), of unknown length. So anchor on
    the output's own boundary run, find it in the neighbour, and confirm the
    overlap continues all the way to the join. Simply searching for the
    neighbour's text anywhere in the output would flag every span of a
    repetitive document. Whitespace is flattened first so a model that reflows
    lines is still caught.
    """
    edge, text = _flat(edge), _flat(text)
    if len(edge) < _BLEED_WINDOW or len(text) < _BLEED_WINDOW:
        return False
    probe = text[:_BLEED_WINDOW] if at_start else text[-_BLEED_WINDOW:]
    pos = edge.find(probe)
    while pos != -1:
        if at_start and edge[pos:] == text[:len(edge) - pos]:
            return True
        if not at_start and edge[:pos + _BLEED_WINDOW] == text[-(pos + _BLEED_WINDOW):]:
            return True
        pos = edge.find(probe, pos + 1)
    return False


def check_output(job: dict, index: int, text: str,
                 finish_reason: str | None) -> list[str]:
    """Reasons this rewrite should not be auto-accepted."""
    spans = job["spans"]
    original = spans[index]["original"]
    flags = []
    if finish_reason == "length":
        flags.append("truncated")
    if not text.strip():
        flags.append("empty")
        return flags
    deleting = normalize_settings(job.get("settings"))["allow_deletions"]
    # Edges are restored mechanically, but a paragraph break *inside* a span is
    # the model's to keep. Losing one silently merges two paragraphs, so surface
    # it for review rather than forcing it back — reflowing a line can be a fair
    # edit, and only the reviewer can tell the two apart. Moot once the job
    # allows removals: taking out content takes out the blank lines around it.
    if not deleting and text.count("\n\n") < original.count("\n\n"):
        flags.append("lost-break")
    # Catches the failure the length checks miss: a fluent replacement of about
    # the right size that keeps none of the author's words. Creative models slip
    # into this readily, especially on a short span.
    if drawn_from_source(original, text) < MIN_DRAWN_FROM_SOURCE:
        flags.append("invented")
    if not reaches_end(original, text):
        flags.append("stops-short")
    # Skipped on very short spans (a trailing fragment, say), where one added
    # word swings the ratio past the threshold on its own.
    if len(original) >= _RATIO_FLOOR:
        floor = MIN_RATIO_DELETING if deleting else MIN_RATIO
        ratio = len(text) / len(original)
        if ratio < floor or ratio > MAX_RATIO:
            flags.append("length-ratio")
    # Only worth testing when the rewrite actually grew: bleeding adds the
    # neighbour's text, whereas a clean rewrite stays about the original's
    # length. The gate also keeps perfectly periodic prose — where a bleed reads
    # identically to correct continuation — from flagging every span.
    if len(_flat(text)) - len(_flat(original)) >= _BLEED_WINDOW:
        start, end = _window(job, index)
        before = "".join(current_text(s) for s in spans[start:index])
        after = "".join(s["original"] for s in spans[index + 1:end + 1])
        if _bleeds(before, text, at_start=True) or _bleeds(after, text, at_start=False):
            flags.append("context-bleed")
    return flags


# ── Generation ──────────────────────────────────────────────────────────────

def busy() -> bool:
    """True when the single llama-server is already generating something."""
    run = appstate.state.editing
    if run is not None and not run.get("done"):
        return True
    return bool(appstate.state.generations)


def oversized(job: dict, index: int) -> bool:
    """Exact token check for one span, run just before it is sent.

    Packing uses a character estimate, so a span of dense text can come out over
    the reply cap; splitting it here is cheaper than burning a truncated
    generation to find out.
    """
    count, exact = engine.server.count_tokens(job["spans"][index]["original"])
    return exact and count > job["budgets"]["span_target"]


def start_span(job: dict, index: int, nudge: str = "") -> dict:
    """Stream one span's rewrite on a worker thread; returns the run state."""
    messages = build_messages(job, index, nudge)
    params = sampler_params(normalize_settings(job.get("settings")))
    run = {
        "job_id": job["id"],
        "index": index,
        "text": "",
        "reasoning": "",
        "finish_reason": None,
        "done": False,
        "error": None,
    }
    appstate.state.editing = run

    def worker():
        try:
            for event in engine.server.stream_chat(messages, params):
                kind = event.get("type")
                if kind == "finish":
                    run["finish_reason"] = event.get("reason")
                    continue
                delta = event.get("text", "")
                if not delta:
                    continue
                if kind == "reasoning":
                    run["reasoning"] += delta
                else:
                    run["text"] += delta
        except Exception as exc:  # noqa: BLE001 - surfaced to the observing page
            run["error"] = str(exc)
        finally:
            run["done"] = True

    threading.Thread(target=worker, daemon=True).start()
    return run


def record_result(job: dict, run: dict) -> dict:
    """Fold a finished run into its span and persist the job."""
    span = job["spans"][run["index"]]
    span["attempts"] += 1
    span["finish_reason"] = run.get("finish_reason")
    if run.get("error"):
        span["status"] = FLAGGED
        span["flags"] = ["error"]
        span["rewritten"] = ""
        store.save_job(job)
        return span
    text = reattach_edges(span["original"], run["text"])
    flags = check_output(job, run["index"], text, run.get("finish_reason"))
    span["rewritten"] = text
    span["flags"] = flags
    span["status"] = FLAGGED if flags else PROPOSED
    store.save_job(job)
    return span


def decide(job: dict, index: int, status: str, text: str | None = None) -> None:
    """Accept, hand-edit, or reject one span's rewrite."""
    span = job["spans"][index]
    if text is not None:
        span["rewritten"] = reattach_edges(span["original"], text)
    span["status"] = status
    store.save_job(job)


def next_pending(job: dict, after: int = -1) -> int | None:
    for i in range(after + 1, len(job["spans"])):
        if job["spans"][i]["status"] in (PENDING, PROPOSED, FLAGGED):
            return i
    return None


def progress(job: dict) -> tuple[int, int]:
    """How much of the document is decided, in source characters.

    Not in spans: a span whose reply hits the token cap is halved and retried, so
    the span count climbs part-way through a run. Counting spans makes the total
    grow under the reader and the bar slide backwards, while the document itself
    never changed length.
    """
    spans = job["spans"]
    if not spans:
        return 0, 0
    done = sum(s["end"] - s["start"] for s in spans
               if s["status"] in (ACCEPTED, ORIGINAL))
    return done, spans[-1]["end"] - spans[0]["start"]


# ── Job lifecycle ───────────────────────────────────────────────────────────

def prepare(job: dict) -> dict:
    """Attach derived budgets and, on first open, the span list."""
    settings = normalize_settings(job.get("settings"))
    job["settings"] = settings
    system_tokens = est_tokens(job.get("system_prompt") or "")
    job["budgets"] = budgets(settings, engine.server.context_limit(), system_tokens)
    if not job.get("spans"):
        job["spans"] = make_spans(job["source_text"], job["budgets"]["span_target"])
        store.save_job(job)
    return job


def create(title: str, source_text: str, system_prompt: str,
           settings: dict) -> dict:
    job = store.new_job(title, source_text, system_prompt,
                        engine.server.model or appstate.state.current_model,
                        normalize_settings(settings))
    return prepare(job)
