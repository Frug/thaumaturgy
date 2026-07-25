"""Span-by-span document editing.

A model capped at a few hundred output tokens cannot rewrite a long document in
one reply, so the loop is inverted: a cursor walks the document and each
generation rewrites one bounded span, with the surrounding text supplied as
read-only context. Accepted rewrites are spliced back in and become context for
the spans below them.

Two levels, both of which earn their place:

  * Chunks anchor the *start* of the prompt body for a run of spans. Splicing
    invalidates llama.cpp's KV cache from the splice point onward, but the
    splice point only moves forward — so as long as the body's start is pinned,
    everything above the cursor stays cached. A window that slid every span
    would reset the cache every span.
  * Spans are the unit the model actually rewrites, sized to fit the reply cap.

No NiceGUI here: the page is presentation only.
"""

import math
import re
import threading

from thaumaturgy import appstate, engine, store

SPAN_OPEN = "<<<SPAN>>>"
SPAN_CLOSE = "<<</SPAN>>>"

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
    "overlap_pct": 0.10,      # lookbehind/lookahead, as a fraction of the body
    "auto_accept_clean": False,
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
    out["auto_accept_clean"] = bool(src.get("auto_accept_clean",
                                            out["auto_accept_clean"]))
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
    # Fold a runt tail back into its neighbour. A span of a few characters costs
    # a whole generation, gives the model almost nothing to work with, and falls
    # under the length-ratio guard's floor — so nonsense comes back unflagged.
    if len(out) > 1 and out[-1][1] - out[-1][0] < MIN_SPAN_CHARS:
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
    body = max(512, usable - span_target - PROMPT_OVERHEAD)
    overlap = int(body * settings["overlap_pct"])
    core = max(span_target, body - 2 * overlap)
    return {"span_target": span_target, "body": body,
            "overlap": overlap, "core": core}


def assign_chunks(spans: list[dict], core_tokens: int) -> list[int]:
    """Map each span to a chunk index; chunks fill `core_tokens` of document.

    The chunk index is what pins the prompt body's start position, which is what
    keeps the cached prefix alive across the spans inside it.
    """
    out, chunk, used = [], 0, 0
    for span in spans:
        size = est_tokens(span["original"])
        if used and used + size > core_tokens:
            chunk += 1
            used = 0
        out.append(chunk)
        used += size
    return out


def current_text(span: dict) -> str:
    """What this span contributes to the document as it stands."""
    return span["rewritten"] if span["status"] == ACCEPTED else span["original"]


def assemble(job: dict) -> str:
    """The document as edited so far — undecided spans fall back to original."""
    return "".join(current_text(s) for s in job["spans"])


def _window(job: dict, index: int) -> tuple[int, int]:
    """First and last span index (inclusive) of the prompt body around `index`.

    Lookbehind reaches back to the start of the span's chunk and then `overlap`
    tokens further; lookahead runs `overlap` tokens past the chunk's end.
    """
    spans = job["spans"]
    chunks = assign_chunks(spans, job["budgets"]["core"])
    overlap = job["budgets"]["overlap"]
    chunk = chunks[index]
    first = next(i for i, c in enumerate(chunks) if c == chunk)
    last = max(i for i, c in enumerate(chunks) if c == chunk)

    start, used = first, 0
    while start > 0 and used + est_tokens(current_text(spans[start - 1])) <= overlap:
        start -= 1
        used += est_tokens(current_text(spans[start]))
    end, used = last, 0
    while end < len(spans) - 1 and used + est_tokens(spans[end + 1]["original"]) <= overlap:
        end += 1
        used += est_tokens(spans[end]["original"])
    return start, end


def build_messages(job: dict, index: int, nudge: str = "") -> list[dict]:
    """Assemble the chat messages that ask for one span's rewrite.

    The span is both marked in place and restated at the tail. The restatement
    is what the model actually copies from — a few hundred tokens back rather
    than buried mid-context, which matters a lot for verbatim fidelity.
    """
    spans = job["spans"]
    start, end = _window(job, index)
    before = "".join(current_text(s) for s in spans[start:index])
    after = "".join(s["original"] for s in spans[index + 1:end + 1])
    target = spans[index]["original"]

    body = f"{before}{SPAN_OPEN}{target}{SPAN_CLOSE}{after}"
    instruction = (
        "Rewrite the marked span. Output only the corrected passage — no "
        "commentary, no markers, and none of the surrounding text."
    )
    if nudge.strip():
        instruction += f"\n\nAdditional instruction: {nudge.strip()}"
    user = (
        f"Document region for context:\n\n{body}\n\n"
        f"{instruction}\n\nThe span to rewrite:\n\n{target}"
    )
    system = (job.get("system_prompt") or "").strip()
    if not system:
        return [{"role": "user", "content": user}]
    if engine.server.supports_system_role():
        return [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    return [{"role": "user", "content": f"{system}\n\n{user}"}]


# ── Output guards ───────────────────────────────────────────────────────────

MIN_RATIO, MAX_RATIO = 0.5, 2.0
_RATIO_FLOOR = 40
_BLEED_WINDOW = 60


def _flat(text: str) -> str:
    return " ".join(text.split())


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
    if SPAN_OPEN in text or SPAN_CLOSE in text:
        flags.append("markers")
    # Edges are restored mechanically, but a paragraph break *inside* a span is
    # the model's to keep. Losing one silently merges two paragraphs, so surface
    # it for review rather than forcing it back — reflowing a line can be a fair
    # edit, and only the reviewer can tell the two apart.
    if text.count("\n\n") < original.count("\n\n"):
        flags.append("lost-break")
    # Skipped on very short spans (a trailing fragment, say), where one added
    # word swings the ratio past the threshold on its own.
    if len(original) >= _RATIO_FLOOR:
        ratio = len(text) / len(original)
        if ratio < MIN_RATIO or ratio > MAX_RATIO:
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
    decided = sum(1 for s in job["spans"] if s["status"] in (ACCEPTED, ORIGINAL))
    return decided, len(job["spans"])


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
