"""Persistence for chats, scenarios, and model/generation settings."""

import json
import os
import time
from pathlib import Path

import yaml

from thaumaturgy.paths import chats_dir, data_dir, sub_dir


def _app_config_path() -> Path:
    return data_dir() / "app_config.yaml"


def _as_mapping(value) -> dict:
    """A dict, or an empty one; these files are hand-editable."""
    return value if isinstance(value, dict) else {}


def _as_list(value) -> list:
    """A list, or an empty one; a bare string is not a list of names."""
    return value if isinstance(value, list) else []


def _write_atomic(path: Path, text: str) -> None:
    """Replace `path` in one step, so a concurrent reader never sees a partial file.

    Chats are saved from the generation worker thread while the UI thread lists
    and loads them, and a plain write would expose the truncated intermediate.
    """
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _chat_group_dir_name(scenario: str | None) -> str:
    name = scenario or ""
    safe = "".join(c if (c.isalnum() or c in " .-_") else "_" for c in name).strip(" .")
    return safe or "unknown_scenario"


def _chat_path(chat_id: str, scenario: str | None):
    return chats_dir() / _chat_group_dir_name(scenario) / f"{chat_id}.json"


def _chat_paths(chat_id: str) -> list[Path]:
    """Every file holding this chat: its scenario dir, plus any stale location.

    rglob covers chats_dir() itself, so ungrouped legacy chats are found too.
    """
    return sorted(chats_dir().rglob(f"{chat_id}.json"))


def _find_chat_path(chat_id: str) -> Path | None:
    paths = _chat_paths(chat_id)
    return paths[0] if paths else None


def _all_chat_files() -> list[Path]:
    return sorted(chats_dir().rglob("*.json"))


def _title_from(messages: list[dict]) -> str:
    for m in messages:
        if m.get("role") == "user" and m.get("text", "").strip():
            t = m["text"].strip().splitlines()[0]
            return (t[:40] + "…") if len(t) > 40 else t
    return "New chat"


def new_chat(scenario: str | None, model: str | None,
             opening_text: str | None = None) -> dict:
    now = time.time()
    chat_id = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
    # Avoid collisions if two chats start within the same second.
    while _find_chat_path(chat_id):
        now += 1
        chat_id = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
    chat = {
        "id": chat_id,
        "scenario": scenario,
        "model": model,
        "title": "New chat",
        "created": now,
        "updated": now,
        "messages": [],
    }
    if opening_text:
        chat["messages"].append({"role": "assistant", "name": scenario, "text": opening_text})
    save_chat(chat)
    return chat


def save_chat(chat: dict) -> None:
    chat["updated"] = time.time()
    if not chat.get("title_custom"):
        chat["title"] = _title_from(chat.get("messages", []))
    _write_chat(chat)


def _write_chat(chat: dict) -> None:
    target = _chat_path(chat["id"], chat.get("scenario"))
    # Stale copies only exist right after a chat moves, and this runs twice a
    # second while streaming, so skip the tree walk once it's settled.
    settled = target.exists()
    if not settled:
        target.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(target,
                  json.dumps(chat, indent=2, ensure_ascii=False))
    if not settled:
        for old in _chat_paths(chat["id"]):
            if old != target:
                old.unlink(missing_ok=True)


def load_chat(chat_id: str) -> dict | None:
    p = _find_chat_path(chat_id)
    if p is None:
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def list_chats(scenario: str | None = None) -> list[dict]:
    out = []
    for p in _all_chat_files():
        try:
            c = json.loads(p.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if scenario is None or c.get("scenario") == scenario:
            out.append(c)
    out.sort(key=lambda c: c.get("updated", 0), reverse=True)
    return out


def rename_chat(chat_id: str, title: str) -> bool:
    """Give a chat a title of its own, one save_chat won't derive away.

    Leaves "updated" alone: a rename shouldn't reorder the sidebar.
    """
    chat = load_chat(chat_id)
    title = (title or "").strip()
    if chat is None or not title:
        return False
    chat["title"] = title
    chat["title_custom"] = True
    _write_chat(chat)
    return True


def delete_chat(chat_id: str) -> None:
    for p in _chat_paths(chat_id):
        p.unlink(missing_ok=True)


# ── Editing jobs (one JSON file each under <data>/editing/) ─────────────────
# A job holds the source document, the span decisions made so far, and the
# settings the run was started with. Saved after every decision so a long run
# survives a restart.

def editing_dir():
    return sub_dir("editing")


def _job_path(job_id: str) -> Path:
    return editing_dir() / f"{job_id}.json"


def new_job(title: str, source_text: str, system_prompt: str,
            model: str | None, settings: dict) -> dict:
    now = time.time()
    job_id = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
    while _job_path(job_id).exists():
        now += 1
        job_id = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
    job = {
        "id": job_id,
        "title": title or "Untitled document",
        "created": now,
        "updated": now,
        "model": model,
        "system_prompt": system_prompt,
        "settings": dict(settings),
        "source_text": source_text,
        "spans": [],
    }
    save_job(job)
    return job


def save_job(job: dict) -> None:
    job["updated"] = time.time()
    _write_atomic(_job_path(job["id"]),
                  json.dumps(job, indent=2, ensure_ascii=False))


def load_job(job_id: str) -> dict | None:
    p = _job_path(job_id)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def list_jobs() -> list[dict]:
    out = []
    for p in sorted(editing_dir().glob("*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except (ValueError, OSError):
            continue
    out.sort(key=lambda j: j.get("updated", 0), reverse=True)
    return out


def delete_job(job_id: str) -> None:
    _job_path(job_id).unlink(missing_ok=True)


# ── Editing instruction sets (one file: <data>/editing_prompts.yaml) ────────
# The prompt text for an editing job: the author's own instructions plus the
# wrapper the page puts around each passage. Kept apart from a job so a set of
# instructions that works can be reused on the next document. Deliberately dumb
# here: the defaults live with the code that uses them, in thaumaturgy.editing.

def _edit_prompts_path() -> Path:
    return data_dir() / "editing_prompts.yaml"


def load_edit_prompts() -> dict:
    """Saved instruction sets by name; empty on first run or a bad file."""
    try:
        doc = yaml.safe_load(_edit_prompts_path().read_text(encoding="utf-8")) or {}
    except (yaml.YAMLError, OSError):
        return {}
    sets = _as_mapping(_as_mapping(doc).get("sets"))
    return {name: _as_mapping(vals) for name, vals in sets.items()
            if isinstance(name, str) and name.strip()}


def save_edit_prompts(sets: dict) -> None:
    _write_atomic(_edit_prompts_path(),
                  yaml.safe_dump({"sets": sets}, sort_keys=True, allow_unicode=True))


# ── App config ──────────────────────────────────────────────────────────────

def load_app_config() -> dict:
    p = _app_config_path()
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except (yaml.YAMLError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def save_app_config(config: dict) -> None:
    _write_atomic(_app_config_path(),
                  yaml.safe_dump(config, sort_keys=False, allow_unicode=True))


def save_last_loaded_model(model_name: str | None) -> None:
    config = load_app_config()
    if model_name:
        config["last_loaded_model"] = model_name
    else:
        config.pop("last_loaded_model", None)
    save_app_config(config)


def last_loaded_model() -> str | None:
    model = load_app_config().get("last_loaded_model")
    return model if isinstance(model, str) and model else None


def save_last_scenario(name: str | None) -> None:
    config = load_app_config()
    if name:
        config["last_scenario"] = name
    else:
        config.pop("last_scenario", None)
    save_app_config(config)


def last_scenario() -> str | None:
    name = load_app_config().get("last_scenario")
    return name if isinstance(name, str) and name else None


def log_dir_setting() -> str:
    """The configured diagnostic-log directory, or "" when logging is off."""
    value = load_app_config().get("log_dir")
    return value.strip() if isinstance(value, str) else ""


def save_log_dir(path: str | None) -> None:
    from thaumaturgy import paths

    config = load_app_config()
    path = (path or "").strip()
    if path:
        config["log_dir"] = path
    else:
        config.pop("log_dir", None)
    save_app_config(config)
    paths.reset_log_dir()


def compaction_divider() -> bool:
    """Whether the chat marks where its recap takes over. On unless turned off."""
    value = load_app_config().get("compaction_divider")
    return True if value is None else bool(value)


def save_compaction_divider(show: bool) -> None:
    config = load_app_config()
    config["compaction_divider"] = bool(show)
    save_app_config(config)


# How a fold is summarized: in one generation, or one per span of it.
COMPACTION_STRATEGIES = ("single", "passes")
DEFAULT_COMPACTION_STRATEGY = "single"


def compaction_strategy() -> str:
    value = load_app_config().get("compaction_strategy")
    return value if value in COMPACTION_STRATEGIES else DEFAULT_COMPACTION_STRATEGY


def save_compaction_strategy(strategy: str) -> None:
    config = load_app_config()
    config["compaction_strategy"] = (
        strategy if strategy in COMPACTION_STRATEGIES else DEFAULT_COMPACTION_STRATEGY)
    save_app_config(config)


# ── Compaction prompt (<data>/compaction.yaml) ──────────────────────────────
# How a chat's older turns get condensed once it outgrows the context window.
# Seeded on first read so it can be edited like any other data file.

def _compaction_path() -> Path:
    return data_dir() / "compaction.yaml"


def default_compaction_prompt() -> dict:
    return {
        "system": (
            "You are an archivist. You condense an ongoing conversation so it "
            "can continue after its earliest turns have scrolled out of view. "
            "What you leave out is lost: the original turns are never consulted "
            "again. You never continue the conversation, never speak as any "
            "participant, and never address the reader."
        ),
        "instruction": (
            "Condense the {turns} turns below into a record the conversation "
            "can be continued from. They will not be available again, so "
            "anything you omit is gone.\n\n"
            "Write {min_words}-{max_words} words, under each of these headings "
            "that applies:\n\n"
            "**Participants** - who is taking part, including any personas or "
            "characters, and what has been established about them.\n"
            "**What happened** - what was said, done, and worked out, in the "
            "order it occurred, from the start of this span through to its "
            "end. Cover the whole span evenly; do not hurry through the early "
            "parts to reach the recent ones.\n"
            "**Where things stand** - the situation as of the last turn, "
            "including anything left in progress.\n"
            "**Open threads** - questions asked and unanswered, requests and "
            "commitments not yet met, and what each participant intends to do "
            "next.\n"
            "**Details to preserve** - specifics a later turn would contradict "
            "if they were forgotten: names, numbers, definitions, stated "
            "preferences, established facts, decisions already settled, and "
            "any wording worth keeping verbatim.\n\n"
            "The conversation may be of any kind - a discussion, work being "
            "carried out, a story, something else - so record what actually "
            "took place rather than fitting it to a form. Prefer specifics to "
            "summary: a name, a number, or a quoted phrase is worth more than "
            "a sentence describing it in general terms. Err long, since length "
            "costs little here and detail lost cannot be recovered.\n\n"
            "{recap}\n\n"
            "Transcript:\n{transcript}"
        ),
        "carry": (
            "Recap of earlier events, already condensed once. It is now the "
            "only record of them, so carry everything from it that still "
            "matters into the new recap:"
        ),
    }


# Earlier defaults, replaced on load when the file still holds them verbatim.
# An edited file is the user's own and is never touched.
_SUPERSEDED_COMPACTION = {
    "system": [
        "You are an archivist. You condense an ongoing story so it can "
        "continue after its earliest scenes have scrolled out of view. You "
        "never continue the story, never speak as a character, and never "
        "address the reader."
    ],
    "instruction": [
        "Condense the {turns} turns below into a record the conversation "
        "can be continued from. They will not be available again, so "
        "anything you omit is gone.\n\n"
        "Write {min_words}-{max_words} words, under each of these headings "
        "that applies:\n\n"
        "**Participants** - everyone named, who they are, and how they "
        "stand with each other now.\n"
        "**What happened** - events and exchanges in the order they "
        "occurred, from the start of this span through to its end. Cover "
        "the whole span evenly; do not hurry through the early parts to "
        "reach the recent ones.\n"
        "**Where things stand** - the situation as of the last turn: "
        "setting, state, and anything left in progress.\n"
        "**Open threads** - questions raised and unanswered, commitments "
        "made and unmet, plans stated and not yet carried out, and what "
        "each participant currently intends.\n"
        "**Details to preserve** - specifics a later turn would contradict "
        "if they were forgotten: names, numbers, descriptions, established "
        "facts, decisions already settled, and any distinctive phrasing "
        "worth keeping.\n\n"
        "Prefer specifics to summary. A name, a number, or a quoted phrase "
        "is worth more than a sentence describing it in general terms. Err "
        "long: length costs little here, and detail lost cannot be "
        "recovered.\n\n"
        "{recap}\n\n"
        "Transcript:\n{transcript}",
        "Write a recap of the story so far, in at most {max_words} words.\n\n"
        "Keep: who the characters are and how they stand with each other, "
        "what happened and in what order, decisions taken and promises "
        "given, details of the setting that later scenes depend on, and "
        "every thread left unresolved.\n"
        "Drop: turn-by-turn phrasing, pleasantries, and description that no "
        "longer bears on anything.\n\n"
        "Write continuous past-tense prose, third person, with no headings "
        "and no bullet points. Use names exactly as they appear.\n\n"
        "{recap}\n\n"
        "Transcript:\n{transcript}"
    ],
    "carry": [
        "Recap of earlier events, already condensed once. Carry what still "
        "matters into the new recap:"
    ],
}


def load_compaction_prompt() -> dict:
    """The recap instructions, seeded on disk the first time they are needed.

    A field still holding an older default is replaced by the current one and
    written back, so an untouched file keeps up. Anything edited is kept as is.
    """
    doc = default_compaction_prompt()
    try:
        saved = yaml.safe_load(_compaction_path().read_text(encoding="utf-8"))
    except (yaml.YAMLError, OSError):
        saved = None
    if not isinstance(saved, dict):
        save_compaction_prompt(doc)
        return doc
    upgraded = False
    for key in doc:
        value = saved.get(key)
        if not (isinstance(value, str) and value.strip()):
            upgraded = True          # missing field: the default fills it in
        elif value.strip() in [old.strip() for old in _SUPERSEDED_COMPACTION[key]]:
            upgraded = True
        else:
            doc[key] = value
    if upgraded:
        save_compaction_prompt(doc)
    return doc


def save_compaction_prompt(doc: dict) -> None:
    _write_atomic(_compaction_path(),
                  yaml.safe_dump(doc, sort_keys=False, allow_unicode=True,
                                 default_flow_style=False))


# ── Scenarios (one YAML file each under <data>/scenarios/) ──────────────────
# A scenario dict carries a "_file" key (its on-disk slug) so renames can move
# the file.

DEFAULT_SCENARIOS_DIR = Path(__file__).parent / "defaults" / "scenarios"
SCENARIO_SEED_MARKER = ".defaults_seeded"


def scenarios_dir():
    return sub_dir("scenarios")


def _as_variables(value) -> dict:
    """A scenario's {{name}} -> value pairs, as strings on both sides.

    These files are hand-editable, so a number or a date written unquoted comes
    back as one; it stands for text in the scenario either way. A name with
    nothing to put in the braces is dropped.
    """
    out = {}
    for name, val in _as_mapping(value).items():
        name = str(name).strip()
        if name and not isinstance(val, (dict, list)):
            out[name] = "" if val is None else str(val)
    return out


def _slug(name: str) -> str:
    keep = "".join(c if (c.isalnum() or c in " -_") else "_" for c in (name or "")).strip()
    return keep or "unnamed"


def list_scenarios() -> list[dict]:
    d = scenarios_dir()
    seed_marker = d / SCENARIO_SEED_MARKER
    if not seed_marker.exists():
        if not any(d.glob("*.yaml")):
            for p in sorted(DEFAULT_SCENARIOS_DIR.glob("*.yaml")):
                target = d / p.name
                if not target.exists():
                    target.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
        seed_marker.write_text("Default scenarios seeded.\n", encoding="utf-8")
    out = []
    for p in sorted(d.glob("*.yaml")):
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except (yaml.YAMLError, OSError):
            continue
        out.append({
            "name": data.get("name", p.stem),
            "context": data.get("context", ""),
            "opening_text": data.get("opening_text", ""),
            "variables": _as_variables(data.get("variables")),
            "_file": p.stem,
        })
    out.sort(key=lambda s: s["name"].lower())
    return out


def save_scenario(scenario: dict) -> None:
    new_slug = _slug(scenario.get("name", ""))
    old_file = scenario.get("_file")
    if old_file and old_file != new_slug:
        old = scenarios_dir() / f"{old_file}.yaml"
        if old.exists():
            old.unlink()
    data = {
        "name": scenario.get("name", ""),
        "context": scenario.get("context", ""),
        "opening_text": scenario.get("opening_text", ""),
    }
    variables = _as_variables(scenario.get("variables"))
    if variables:  # left out entirely rather than written as an empty mapping
        data["variables"] = variables
    (scenarios_dir() / f"{new_slug}.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    scenario["_file"] = new_slug


def delete_scenario(scenario: dict) -> None:
    slug = scenario.get("_file") or _slug(scenario.get("name", ""))
    p = scenarios_dir() / f"{slug}.yaml"
    if p.exists():
        p.unlink()


# ── Parameter sets (persisted as one file: <data>/presets.yaml) ──────────────
# Kept unified in a single file (not one-per-set) so the whole collection is
# trivial to gitignore. On first run the file is seeded from BUILTIN_PRESETS;
# thereafter it's the user's own; edits/renames/deletes all land here.

BUILTIN_PRESETS = {
    "Default": dict(max_new_tokens=512, temperature=0.8, top_p=0.95, top_k=40, min_p=0.05, repetition_penalty=1.10, recap_tokens=4000),
    "Creative": dict(max_new_tokens=512, temperature=1.10, top_p=0.98, top_k=100, min_p=0.02, repetition_penalty=1.05, recap_tokens=4000),
    "Precise": dict(max_new_tokens=512, temperature=0.40, top_p=0.90, top_k=20, min_p=0.10, repetition_penalty=1.15, recap_tokens=4000),
    "Deterministic": dict(max_new_tokens=512, temperature=0.00, top_p=1.00, top_k=1, min_p=0.00, repetition_penalty=1.00, recap_tokens=4000),
}
DEFAULT_PRESET = "Default"
CUSTOM = "Custom"


def _presets_path():
    return data_dir() / "presets.yaml"


def _default_presets_doc() -> dict:
    sets = {name: dict(vals) for name, vals in BUILTIN_PRESETS.items()}
    sets[CUSTOM] = dict(BUILTIN_PRESETS[DEFAULT_PRESET])
    return {"sets": sets, "order": [*BUILTIN_PRESETS, CUSTOM], "model_defaults": {}}


def load_presets() -> dict:
    """Return {sets, order, model_defaults}, seeding defaults on first run.

    Resilient to hand-edits: a missing/corrupt file falls back to defaults, and
    order/model_defaults are reconciled against the sets actually present.
    """
    p = _presets_path()
    if not p.exists():
        doc = _default_presets_doc()
        save_presets(doc)
        return doc
    try:
        doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except (yaml.YAMLError, OSError):
        return _default_presets_doc()
    sets = doc.get("sets") or {}
    if not sets:
        return _default_presets_doc()
    order = [n for n in (doc.get("order") or list(sets)) if n in sets]
    order += [n for n in sets if n not in order]
    model_defaults = {m: s for m, s in (doc.get("model_defaults") or {}).items() if s in sets}
    return {"sets": sets, "order": order, "model_defaults": model_defaults}


def save_presets(doc: dict) -> None:
    _write_atomic(_presets_path(),
                  yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))


# ── Runtime settings (model loading settings) ───────────────────────────────
# The llama-server launch settings: GPU layers, requested context size, KV-cache
# type, chat template override, and llama.cpp reasoning controls.
#
#
# Each model owns its settings: the values are bounded by model properties
# (block count, trained context, template), so the same numbers mean something
# different on another model. Named sets are templates: starting points copied
# onto a model, not a live binding.

# Injected as the model's own last thought before the forced end-of-thinking
# tag, so it has to read in its voice. Wording follows vLLM's; llama.cpp ships
# no default. Only the graceful-vs-abrupt handoff is well evidenced, not this
# phrasing over another.
DEFAULT_REASONING_BUDGET_MESSAGE = "Let me stop thinking and answer now."

DEFAULT_CACHE_RAM = 0
DEFAULT_CTX_CHECKPOINTS = 2
# One slot: llama.cpp splits -c evenly across slots. Each slot takes an even
# portion of context size. With one slot, a reply in a second chat waits for the
# first to finish instead of streaming alongside it.
DEFAULT_PARALLEL_SLOTS = 1

BUILTIN_RUNTIME_TEMPLATES = {
    "Default": dict(
        gpu_layers=-1,
        context_size=0,
        cache_type="fp16",
        chat_template="auto",
        reasoning="auto",
        reasoning_budget=-1,
        reasoning_budget_message=DEFAULT_REASONING_BUDGET_MESSAGE,
        cache_ram=DEFAULT_CACHE_RAM,
        ctx_checkpoints=DEFAULT_CTX_CHECKPOINTS,
    ),
}
DEFAULT_RUNTIME_TEMPLATE = "Default"
_OLD_RUNTIME_DEFAULT = dict(gpu_layers=100, context_size=8192, cache_type="fp16")

RUNTIME_CACHE_TYPES = ("fp16", "q8_0", "q4_0")
RUNTIME_CHAT_TEMPLATES = ("auto", "gemma")
RUNTIME_REASONING_MODES = ("auto", "off", "on")


def _runtime_settings_path():
    return data_dir() / "runtime_settings.yaml"


def _runtime_profiles_path():
    """Pre-per-model layout; read once to migrate, never written again."""
    return data_dir() / "runtime_profiles.yaml"


def normalize_runtime(vals) -> dict:
    """Coerce anything into a full settings dict; the YAML is hand-editable."""
    src = vals if isinstance(vals, dict) else {}
    out = dict(BUILTIN_RUNTIME_TEMPLATES[DEFAULT_RUNTIME_TEMPLATE])
    for key, cast in (("gpu_layers", int), ("context_size", int),
                      ("reasoning_budget", int), ("cache_ram", int),
                      ("ctx_checkpoints", int)):
        try:
            out[key] = cast(src.get(key, out[key]))
        except (TypeError, ValueError):
            pass
    out["cache_ram"] = max(-1, out["cache_ram"])
    out["ctx_checkpoints"] = max(0, out["ctx_checkpoints"])
    for key, allowed in (("cache_type", RUNTIME_CACHE_TYPES),
                         ("chat_template", RUNTIME_CHAT_TEMPLATES),
                         ("reasoning", RUNTIME_REASONING_MODES)):
        if src.get(key) in allowed:
            out[key] = src[key]
    # `or`, not a get default: a null in the YAML would stringify to "None"
    # and be handed to llama-server as the model's forced last thought.
    out["reasoning_budget_message"] = str(
        src.get("reasoning_budget_message") or out["reasoning_budget_message"])
    return out


def _default_runtime_doc() -> dict:
    templates = {name: dict(vals) for name, vals in BUILTIN_RUNTIME_TEMPLATES.items()}
    return {"templates": templates, "order": [*BUILTIN_RUNTIME_TEMPLATES], "models": {}}


def _migrate_runtime_profiles() -> dict | None:
    """Convert the old profile-per-model-pin layout into per-model settings.

    A pinned profile's values become the model's own; profiles carry over as
    templates. The old file is left on disk untouched.
    """
    p = _runtime_profiles_path()
    if not p.exists():
        return None
    try:
        doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except (yaml.YAMLError, OSError):
        return None
    sets = _as_mapping(doc).get("sets")
    sets = _as_mapping(sets)
    if not sets:
        return None
    # Custom resolves for the models pinned to it, but doesn't survive as a
    # template; dropping it here would silently reset those models.
    resolved = {}
    for name, vals in sets.items():
        if name in {DEFAULT_RUNTIME_TEMPLATE, CUSTOM} and vals == _OLD_RUNTIME_DEFAULT:
            vals = BUILTIN_RUNTIME_TEMPLATES[DEFAULT_RUNTIME_TEMPLATE]
        resolved[name] = normalize_runtime(vals)
    templates = {n: v for n, v in resolved.items() if n != CUSTOM}
    order = [n for n in (_as_list(doc.get("order")) or list(templates))
             if isinstance(n, str) and n in templates]
    order += [n for n in templates if n not in order]
    models = {m: dict(resolved[s])
              for m, s in _as_mapping(doc.get("model_defaults")).items()
              if isinstance(s, str) and s in resolved}
    if not templates:
        if not models:
            return None
        templates = {name: dict(vals) for name, vals in BUILTIN_RUNTIME_TEMPLATES.items()}
        order = [*templates]
    return {"templates": templates, "order": order, "models": models}


def load_runtime_settings() -> dict:
    """Return {templates, order, models}, migrating or seeding on first run."""
    p = _runtime_settings_path()
    if not p.exists():
        doc = _migrate_runtime_profiles() or _default_runtime_doc()
        save_runtime_settings(doc)
        return doc
    try:
        doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except (yaml.YAMLError, OSError):
        return _default_runtime_doc()
    doc = _as_mapping(doc)
    templates = {n: normalize_runtime(v)
                 for n, v in _as_mapping(doc.get("templates")).items()}
    if not templates:
        templates = {name: dict(vals) for name, vals in BUILTIN_RUNTIME_TEMPLATES.items()}
    order = [n for n in (_as_list(doc.get("order")) or list(templates))
             if isinstance(n, str) and n in templates]
    order += [n for n in templates if n not in order]
    models = {m: normalize_runtime(v) for m, v in _as_mapping(doc.get("models")).items()}
    return {"templates": templates, "order": order, "models": models}


def save_runtime_settings(doc: dict) -> None:
    # Atomic: every editor control writes on change, so a slider drag rewrites
    # this file continuously and a torn write would lose every model.
    _write_atomic(_runtime_settings_path(),
                  yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
