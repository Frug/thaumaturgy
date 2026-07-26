"""Editing service: spans, prompts, output guards, and generation.

A model capped at a few hundred output tokens cannot rewrite a long document in
one reply, so a cursor walks the document and each generation rewrites one
bounded span. Surrounding text goes in an earlier turn; the passage arrives
alone in the final one — otherwise the model starts reproducing at the top of
the context blob. Keep `overlap` low for the same reason; 0 is most faithful.
"""

import json
import math
import re
import threading
import time

from thaumaturgy import appstate, engine, store
from thaumaturgy.editing import validator
from thaumaturgy.editing.validator import drawn_from_source, flatten, reaches_end
from thaumaturgy.paths import log_dir

# Rough bytes-per-token for packing. Spans are verified exactly before send.
CHARS_PER_TOKEN = 4

# Floor so repeated truncation can't shave a span down to nothing.
MIN_SPAN_CHARS = 40

# Headroom for chat template, restated span, and instruction text.
PROMPT_OVERHEAD = 256
SAFETY_RESERVE = 256

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful copy editor. Fix grammar, spelling, and punctuation. "
    "Preserve the author's voice, meaning, and paragraph structure. Do not "
    "add, remove, or reorder content."
)

# Wrapper text around each passage — editable per job, competes with the
# system prompt. Placeholders: {passage}, {before}, {after}, {first_words},
# {last_words}. Missing placeholders are fine (passage appended, context dropped).

DEFAULT_PASSAGE_INSTRUCTION = (
    "Output only the corrected form of the passage below. Reproduce it in "
    "full: your output must begin with \"{first_words}\" and end with "
    "\"{last_words}\". No commentary, no labels, and none of the surrounding "
    "text."
)

EDGE_WORDS = 8  # quoted back to pin the ends of the passage

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

DEFAULT_SETTINGS = {
    "max_new_tokens": 700,
    "temperature": 0.2,
    "top_p": 0.95,
    "top_k": 40,
    "min_p": 0.05,
    "repetition_penalty": 1.05,
    "response_buffer": 150,   # how far under the reply cap to size a span
    # Neighbouring prose is the main cause of wandering; raise carefully.
    "overlap_pct": 0.0,
    "auto_accept_clean": False,
    "allow_deletions": False,
    "passage_instruction": DEFAULT_PASSAGE_INSTRUCTION,
    "context_framing": DEFAULT_CONTEXT_FRAMING,
    "primed_reply": DEFAULT_PRIMED_REPLY,
    "prime_reply": False,
}

PENDING, PROPOSED, ACCEPTED, ORIGINAL, FLAGGED = (
    "pending", "proposed", "accepted", "original", "flagged")

# Saved instruction sets carry task wording, not numeric tuning.
INSTRUCTION_KEYS = ("system_prompt", "passage_instruction", "context_framing",
                    "primed_reply", "prime_reply")


def default_instructions() -> dict:
    return {
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "passage_instruction": DEFAULT_PASSAGE_INSTRUCTION,
        "context_framing": DEFAULT_CONTEXT_FRAMING,
        "primed_reply": DEFAULT_PRIMED_REPLY,
        "prime_reply": False,
    }


def normalize_instructions(vals) -> dict:
    """Coerce a saved set into every key so a hand-edited file can't break the page."""
    src = vals if isinstance(vals, dict) else {}
    out = default_instructions()
    for key in INSTRUCTION_KEYS:
        value = src.get(key)
        if key == "prime_reply":
            out[key] = bool(value) if value is not None else out[key]
        elif isinstance(value, str):
            out[key] = value
    return out


def normalize_settings(vals) -> dict:
    """Coerce anything into a full settings dict; job files are editable."""
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
    for key in ("auto_accept_clean", "allow_deletions", "prime_reply"):
        out[key] = bool(src.get(key, out[key]))
    # Present-but-empty means the author cleared our wording; only fall back
    # when the key is absent or not a string.
    for key in ("passage_instruction", "context_framing", "primed_reply"):
        if isinstance(src.get(key), str):
            out[key] = src[key]
    out["overlap_pct"] = min(0.45, max(0.0, out["overlap_pct"]))
    out["response_buffer"] = max(0, min(out["response_buffer"],
                                        out["max_new_tokens"] - 64))
    return out


def sampler_params(settings: dict) -> dict:
    """Subset of settings that engine.stream_chat consumes."""
    return {k: settings[k] for k in
            ("max_new_tokens", "temperature", "top_p", "top_k", "min_p",
             "repetition_penalty")}


def est_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


# ── Splitting ───────────────────────────────────────────────────────────────
# Spans are contiguous (start, end) offsets. Boundaries land after sentence
# punctuation, at paragraph breaks, and before markdown blocks — never at a
# bare newline (soft wrap in hard-wrapped prose). Opening mid-sentence makes
# the model complete the fragment from context and duplicate words at the join.

_BOUNDARY_RE = re.compile(
    r"[.!?][\"'’”)\]]*\s+"                    # sentence end + following gap
    r"|\n[ \t]*\n\s*"                          # blank line (paragraph break)
    r"|\n(?=[ \t]*(?:[-*+>#]|\d+[.)])[ \t])")  # before a markdown block


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
    # Fold a runt tail into its neighbour. Tiny spans waste a generation and
    # the model treats the fragment as a writing prompt. Skip if merging would
    # overshoot the budget enough to risk truncation.
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
    """Replace one oversize span with two. False once too small to divide usefully."""
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
    # Span appears twice in the prompt: marked in place, and restated at the tail.
    available = max(512, usable - 2 * span_target - PROMPT_OVERHEAD)
    overlap = int(available * settings["overlap_pct"])
    return {"span_target": span_target, "available": available,
            "overlap": overlap}


def current_text(span: dict) -> str:
    """What this span contributes to the document as it stands."""
    return span["rewritten"] if span["status"] == ACCEPTED else span["original"]


def assemble(job: dict) -> str:
    """Document as edited so far — undecided spans fall back to original."""
    return "".join(current_text(s) for s in job["spans"])


def _window(job: dict, index: int) -> tuple[int, int]:
    """First and last span index (inclusive) of prompt context around `index`.

    `overlap` tokens of corrected text behind, original ahead. Wider windows
    bury the passage in similar prose and the model wanders into it.
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
    """Chat messages for one span rewrite.

    Context in an earlier turn; passage alone in the final one. Splicing the
    passage into a single context blob makes the model start at the top of the
    blob. Wrapper text comes from job settings (editable next to the system prompt).
    """
    spans = job["spans"]
    start, end = _window(job, index)
    before = "".join(current_text(s) for s in spans[start:index])
    after = "".join(s["original"] for s in spans[index + 1:end + 1])
    target = spans[index]["original"]

    settings = normalize_settings(job.get("settings"))
    # Opening/closing words turn "the entire passage" into checkable anchors;
    # without them the model often returns only a middle paragraph or the tail.
    words = flatten(target).split()
    instruction = (settings["passage_instruction"]
                   .replace("{first_words}", " ".join(words[:EDGE_WORDS]))
                   .replace("{last_words}", " ".join(words[-EDGE_WORDS:])))
    if nudge.strip():
        instruction = f"{instruction}\n\nAdditional instruction: {nudge.strip()}".strip()
    if "{passage}" in instruction:
        ask = instruction.replace("{passage}", target)
    else:
        ask = f"{instruction}\n\n{target}" if instruction.strip() else target

    turns = []
    framing = settings["context_framing"]
    context = ""
    if (before.strip() or after.strip()) and framing.strip():
        context = framing.replace("{before}", before).replace("{after}", after)
    primed = settings["primed_reply"] if settings["prime_reply"] else ""
    if context and primed.strip():
        turns += [{"role": "user", "content": context},
                  {"role": "assistant", "content": primed}]
    elif context:
        # Chat templates require alternating roles; merge into one user turn.
        ask = f"{context}\n\n{ask}"
    turns.append({"role": "user", "content": ask})

    system = (job.get("system_prompt") or "").strip()
    if not system:
        return turns
    if engine.server.supports_system_role():
        return [{"role": "system", "content": system}, *turns]
    turns[0]["content"] = f"{system}\n\n{turns[0]['content']}"
    return turns


# ── Output guards ───────────────────────────────────────────────────────────

def check_output(job: dict, index: int, text: str,
                 finish_reason: str | None) -> list[str]:
    """Reasons this rewrite should not be auto-accepted.

    Unpacks the job so validator.check can stay pure text-in, reasons-out.
    """
    spans = job["spans"]
    start, end = _window(job, index)
    return validator.check(
        spans[index]["original"], text, finish_reason,
        before="".join(current_text(s) for s in spans[start:index]),
        after="".join(s["original"] for s in spans[index + 1:end + 1]),
        allow_deletions=normalize_settings(job.get("settings"))["allow_deletions"],
    )


def reattach_edges(original: str, text: str) -> str:
    """Restore the span's leading/trailing whitespace from the source.

    Models drop boundary whitespace; a lost newline at a join welds paragraphs.
    """
    lead = original[:len(original) - len(original.lstrip())]
    trail = original[len(original.rstrip()):]
    return lead + text.strip() + trail


# ── Generation ──────────────────────────────────────────────────────────────

def busy() -> bool:
    """True when the single llama-server is already generating something."""
    run = appstate.state.editing
    if run is not None and not run.get("done"):
        return True
    return bool(appstate.state.generations)


def oversized(job: dict, index: int) -> bool:
    """Exact token check just before send. Packing only estimates; split here
    instead of burning a truncated generation."""
    count, exact = engine.server.count_tokens(job["spans"][index]["original"])
    return exact and count > job["budgets"]["span_target"]


def start_span(job: dict, index: int, nudge: str = "") -> dict:
    """Stream one span rewrite on a worker thread; returns the run state."""
    messages = build_messages(job, index, nudge)
    params = sampler_params(normalize_settings(job.get("settings")))
    run = {
        "job_id": job["id"],
        "index": index,
        "messages": messages,  # kept so the page can show what was sent
        "started": time.monotonic(),
        "text": "",
        "reasoning": "",
        "finish_reason": None,
        "done": False,
        "cancelled": False,
        "error": None,
    }
    appstate.state.editing = run

    def worker():
        try:
            for event in engine.server.stream_chat(messages, params):
                if run["cancelled"]:
                    # Leaving the generator closes the stream and frees the slot.
                    break
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


def cancel(run: dict | None) -> None:
    """Ask an in-flight span run to stop at the next streamed token."""
    if run is not None and not run.get("done"):
        run["cancelled"] = True


def _log_span(job: dict, run: dict, span: dict) -> None:
    """Append one attempt to $THAUM_LOG_DIR/editing.jsonl if set. Never raises."""
    directory = log_dir()
    if directory is None:
        return
    try:
        prompt = "".join(m["content"] for m in run.get("messages", []))
        elapsed = max(0.0, time.monotonic() - run.get("started", time.monotonic()))
        out = span["rewritten"]
        record = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "job": job.get("id"),
            "span": run["index"],
            "attempt": span["attempts"],
            "seconds": round(elapsed, 1),
            "prompt_msgs": len(run.get("messages", [])),
            "prompt_tok_est": est_tokens(prompt),
            "overlap_tok": job.get("budgets", {}).get("overlap"),
            "span_chars": len(span["original"]),
            "out_chars": len(out),
            "out_tok_per_s": round(est_tokens(out) / elapsed, 1) if elapsed else None,
            "finish": span["finish_reason"],
            "flags": span["flags"],
            "drawn_from_source": round(drawn_from_source(span["original"], out), 3),
            "reaches_end": reaches_end(span["original"], out),
            "error": run.get("error"),
        }
        with (directory / "editing.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except (OSError, KeyError, TypeError, ValueError):
        pass


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
        _log_span(job, run, span)
        return span
    text = reattach_edges(span["original"], run["text"])
    flags = check_output(job, run["index"], text, run.get("finish_reason"))
    span["rewritten"] = text
    span["flags"] = flags
    span["status"] = FLAGGED if flags else PROPOSED
    store.save_job(job)
    _log_span(job, run, span)
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
    """Decided progress in source characters (not span count).

    Truncated spans are halved and retried, so span count climbs mid-run and a
    span-based bar slides backwards.
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
