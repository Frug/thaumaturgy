"""Persistence for chats — one JSON file per chat under <data>/chats/.

Chats are tied to a character (the list on the chat page filters by character).
No database: each chat is a self-contained JSON document:

    {
      "id": "20260711-143002",
      "character": "Sage Thornwood",
      "model": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
      "title": "...",
      "created": 1783..., "updated": 1783...,
      "messages": [ {"role": "assistant"|"user", "name": "...", "text": "..."} ]
    }
"""

import json
import time

import yaml

from thaumaturgy.paths import chats_dir, data_dir, sub_dir


def _chat_path(chat_id: str):
    return chats_dir() / f"{chat_id}.json"


def _title_from(messages: list[dict]) -> str:
    for m in messages:
        if m.get("role") == "user" and m.get("text", "").strip():
            t = m["text"].strip().splitlines()[0]
            return (t[:40] + "…") if len(t) > 40 else t
    return "New chat"


def new_chat(character: str | None, model: str | None,
             greeting: str | None = None) -> dict:
    now = time.time()
    chat_id = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
    # Avoid collisions if two chats start within the same second.
    while _chat_path(chat_id).exists():
        now += 1
        chat_id = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
    chat = {
        "id": chat_id,
        "character": character,
        "model": model,
        "title": "New chat",
        "created": now,
        "updated": now,
        "messages": [],
    }
    if greeting:
        chat["messages"].append({"role": "assistant", "name": character, "text": greeting})
    save_chat(chat)
    return chat


def save_chat(chat: dict) -> None:
    chat["updated"] = time.time()
    chat["title"] = _title_from(chat.get("messages", []))
    _chat_path(chat["id"]).write_text(
        json.dumps(chat, indent=2, ensure_ascii=False), encoding="utf-8")


def load_chat(chat_id: str) -> dict | None:
    p = _chat_path(chat_id)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def list_chats(character: str | None = None) -> list[dict]:
    out = []
    for p in chats_dir().glob("*.json"):
        try:
            c = json.loads(p.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if character is None or c.get("character") == character:
            out.append(c)
    out.sort(key=lambda c: c.get("updated", 0), reverse=True)
    return out


def delete_chat(chat_id: str) -> None:
    p = _chat_path(chat_id)
    if p.exists():
        p.unlink()


# ── Characters (one YAML file each under <data>/characters/) ─────────────────
# A character dict carries a "_file" key (its on-disk slug) so renames can move
# the file. Format on disk is just name / context / greeting.

DEFAULT_CHARACTERS = [
    {
        "name": "Sage Thornwood",
        "context": ("Sage is an elderly forest hermit and herbalist. Patient, cryptic, "
                    "and endlessly curious about the people who wander into the woods."),
        "greeting": "Ah. The forest whispered that a traveler was near. Sit — the tea is nearly ready.",
    },
    {
        "name": "Unit-7",
        "context": ("Unit-7 is a decommissioned security android rediscovering the world. "
                    "It is literal, polite, and quietly fascinated by human idioms."),
        "greeting": "Greeting acknowledged. I am Unit-7. I am told this is where a conversation begins.",
    },
]


def characters_dir():
    return sub_dir("characters")


def _slug(name: str) -> str:
    keep = "".join(c if (c.isalnum() or c in " -_") else "_" for c in (name or "")).strip()
    return keep or "unnamed"


def list_characters() -> list[dict]:
    d = characters_dir()
    if not any(d.glob("*.yaml")):
        for c in DEFAULT_CHARACTERS:
            save_character(dict(c))
    out = []
    for p in sorted(d.glob("*.yaml")):
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except (yaml.YAMLError, OSError):
            continue
        out.append({
            "name": data.get("name", p.stem),
            "context": data.get("context", ""),
            "greeting": data.get("greeting", ""),
            "_file": p.stem,
        })
    out.sort(key=lambda c: c["name"].lower())
    return out


def save_character(char: dict) -> None:
    new_slug = _slug(char.get("name", ""))
    old_file = char.get("_file")
    if old_file and old_file != new_slug:
        old = characters_dir() / f"{old_file}.yaml"
        if old.exists():
            old.unlink()
    data = {
        "name": char.get("name", ""),
        "context": char.get("context", ""),
        "greeting": char.get("greeting", ""),
    }
    (characters_dir() / f"{new_slug}.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    char["_file"] = new_slug


def delete_character(char: dict) -> None:
    slug = char.get("_file") or _slug(char.get("name", ""))
    p = characters_dir() / f"{slug}.yaml"
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
# GPU layers, requested context size, and KV-cache type.

BUILTIN_RUNTIME_PROFILES = {
    "Default": dict(gpu_layers=-1, context_size=0, cache_type="fp16"),
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
