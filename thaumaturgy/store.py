"""Persistence for chats, scenarios, and model/generation settings."""

import json
import os
import time
from pathlib import Path

import yaml

from thaumaturgy.paths import chats_dir, data_dir, sub_dir


def _app_config_path() -> Path:
    return data_dir() / "app_config.yaml"


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
    chat["title"] = _title_from(chat.get("messages", []))
    target = _chat_path(chat["id"], chat.get("scenario"))
    # Stale copies only exist right after a chat moves, and this runs twice a
    # second while streaming — so skip the tree walk once it's settled.
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


def delete_chat(chat_id: str) -> None:
    for p in _chat_paths(chat_id):
        p.unlink(missing_ok=True)


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


# ── Scenarios (one YAML file each under <data>/scenarios/) ──────────────────
# A scenario dict carries a "_file" key (its on-disk slug) so renames can move
# the file.

DEFAULT_SCENARIOS_DIR = Path(__file__).parent / "defaults" / "scenarios"
SCENARIO_SEED_MARKER = ".defaults_seeded"


def scenarios_dir():
    return sub_dir("scenarios")


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
# thereafter it's the user's own — edits/renames/deletes all land here.

BUILTIN_PRESETS = {
    "Default": dict(max_new_tokens=512, temperature=0.8, top_p=0.95, top_k=40, min_p=0.05, repetition_penalty=1.10),
    "Creative": dict(max_new_tokens=512, temperature=1.10, top_p=0.98, top_k=100, min_p=0.02, repetition_penalty=1.05),
    "Precise": dict(max_new_tokens=512, temperature=0.40, top_p=0.90, top_k=20, min_p=0.10, repetition_penalty=1.15),
    "Deterministic": dict(max_new_tokens=512, temperature=0.00, top_p=1.00, top_k=1, min_p=0.00, repetition_penalty=1.00),
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
    _presets_path().write_text(
        yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")


# ── Runtime profiles (model loading settings) ───────────────────────────────
# These persist the llama-server launch settings that vary by model or machine:
# GPU layers, requested context size, KV-cache type, chat template override, and
# llama.cpp reasoning controls.

# Injected as the model's own last thought before the forced end-of-thinking
# tag, so it has to read in its voice. Wording follows vLLM's; llama.cpp ships
# no default. Only the graceful-vs-abrupt handoff is well evidenced, not this
# phrasing over another.
DEFAULT_REASONING_BUDGET_MESSAGE = "Let me stop thinking and answer now."

BUILTIN_RUNTIME_PROFILES = {
    "Default": dict(
        gpu_layers=-1,
        context_size=0,
        cache_type="fp16",
        chat_template="auto",
        reasoning="auto",
        reasoning_budget=-1,
        reasoning_budget_message=DEFAULT_REASONING_BUDGET_MESSAGE,
    ),
}
DEFAULT_RUNTIME_PROFILE = "Default"
_OLD_RUNTIME_DEFAULT = dict(gpu_layers=100, context_size=8192, cache_type="fp16")


def _runtime_profiles_path():
    return data_dir() / "runtime_profiles.yaml"


def _default_runtime_profiles_doc() -> dict:
    sets = {name: dict(vals) for name, vals in BUILTIN_RUNTIME_PROFILES.items()}
    sets[CUSTOM] = dict(BUILTIN_RUNTIME_PROFILES[DEFAULT_RUNTIME_PROFILE])
    return {"sets": sets, "order": [*BUILTIN_RUNTIME_PROFILES, CUSTOM], "model_defaults": {}}


def load_runtime_profiles() -> dict:
    """Return {sets, order, model_defaults}, seeding defaults on first run."""
    p = _runtime_profiles_path()
    if not p.exists():
        doc = _default_runtime_profiles_doc()
        save_runtime_profiles(doc)
        return doc
    try:
        doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except (yaml.YAMLError, OSError):
        return _default_runtime_profiles_doc()
    sets = doc.get("sets") or {}
    if not sets:
        return _default_runtime_profiles_doc()
    changed = False
    for name, vals in sets.items():
        if name in {DEFAULT_RUNTIME_PROFILE, CUSTOM} and vals == _OLD_RUNTIME_DEFAULT:
            vals.update(BUILTIN_RUNTIME_PROFILES[DEFAULT_RUNTIME_PROFILE])
            changed = True
        before = dict(vals)
        vals.setdefault("gpu_layers", -1)
        vals.setdefault("context_size", 0)
        vals.setdefault("cache_type", "fp16")
        vals.setdefault("chat_template", "auto")
        vals.setdefault("reasoning", "auto")
        vals.setdefault("reasoning_budget", -1)
        vals.setdefault("reasoning_budget_message", DEFAULT_REASONING_BUDGET_MESSAGE)
        changed = changed or vals != before
    order = [n for n in (doc.get("order") or list(sets)) if n in sets]
    order += [n for n in sets if n not in order]
    model_defaults = {m: s for m, s in (doc.get("model_defaults") or {}).items() if s in sets}
    out = {"sets": sets, "order": order, "model_defaults": model_defaults}
    if changed:
        save_runtime_profiles(out)
    return out


def save_runtime_profiles(doc: dict) -> None:
    _runtime_profiles_path().write_text(
        yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
