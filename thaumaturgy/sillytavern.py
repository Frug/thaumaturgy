"""Turn SillyTavern sampler presets into Thaumaturgy parameter sets.

Accepts a text-completion preset (the file under "Text Completion presets"), a
chat-completion preset, or a master export bundling a text-completion preset
with instruct/context templates, a system prompt and a reasoning format. Only
the samplers a parameter set holds are imported; the rest is reported so the
user knows what didn't carry over.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from urllib.parse import unquote, urlsplit

import requests

from thaumaturgy import store

MAX_BYTES = 2 * 1024 * 1024
FETCH_DEADLINE_S = 30

# Preset key -> parameter-set key, per preset format.
_SHARED_KEYS = {"top_p": "top_p", "top_k": "top_k", "min_p": "min_p"}
_TEXTGEN_KEYS = {"genamt": "max_new_tokens", "temp": "temperature",
                 "rep_pen": "repetition_penalty", **_SHARED_KEYS}
_CHAT_KEYS = {"openai_max_tokens": "max_new_tokens", "temperature": "temperature",
              "repetition_penalty": "repetition_penalty", **_SHARED_KEYS}

# Samplers a parameter set can't express, with the value that leaves them off.
# One set to anything else changes generation, so it is worth reporting.
_NEUTRAL = {
    "freq_pen": ("frequency penalty", 0),
    "frequency_penalty": ("frequency penalty", 0),
    "presence_pen": ("presence penalty", 0),
    "presence_penalty": ("presence penalty", 0),
    "typical_p": ("typical-p", 1),
    "tfs": ("TFS", 1),
    "top_a": ("top-a", 0),
    "epsilon_cutoff": ("epsilon cutoff", 0),
    "eta_cutoff": ("eta cutoff", 0),
    "mirostat_mode": ("mirostat", 0),
    "dynatemp": ("dynamic temperature", False),
    "smoothing_factor": ("smoothing", 0),
    "dry_multiplier": ("DRY", 0),
    "xtc_probability": ("XTC", 0),
    "nsigma": ("top-nσ", 0),
    "no_repeat_ngram_size": ("no-repeat n-gram", 0),
    "logit_bias": ("logit bias", []),
    "banned_tokens": ("banned tokens", ""),
    "grammar_string": ("grammar", ""),
}

# Master-export sections beside "preset", and what to call them.
_SECTIONS = {
    "instruct": "instruct template",
    "context": "context template",
    "sysprompt": "system prompt",
    "reasoning": "reasoning format",
}


class PresetFormatError(ValueError):
    """The JSON isn't a SillyTavern preset this importer understands."""


@dataclass
class ImportedPreset:
    name: str
    values: dict
    skipped: list[str] = field(default_factory=list)
    clamped: list[str] = field(default_factory=list)


def _number(raw, key: str) -> float:
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        try:
            if math.isfinite(raw):
                return raw
        except OverflowError:  # An int too large for a float.
            pass
    raise PresetFormatError(f"'{key}' should be a number, not {str(raw)[:40]}")


def _is_neutral(value, neutral) -> bool:
    if value is None or value == neutral:
        return True
    # An empty logit-bias table is as off as an empty list.
    return isinstance(neutral, (list, str)) and not value


def stem_name(path: str) -> str:
    """A file or URL path's last segment, minus a .json suffix of any case."""
    stem = unquote(urlsplit(path).path).rstrip("/").rsplit("/", 1)[-1]
    return stem[:-5] if stem.lower().endswith(".json") else stem


def parse_preset(data, fallback_name: str = "Imported") -> ImportedPreset:
    if not isinstance(data, dict):
        raise PresetFormatError("Expected a JSON object.")
    skipped = [label for key, label in _SECTIONS.items() if data.get(key)]
    preset = data.get("preset") if isinstance(data.get("preset"), dict) else data
    if "temp" in preset or "rep_pen" in preset:
        keys = _TEXTGEN_KEYS
    elif "openai_max_tokens" in preset or "temperature" in preset:
        keys = _CHAT_KEYS
    else:
        raise PresetFormatError("No SillyTavern sampler settings found.")

    values = dict(store.BUILTIN_PRESETS[store.DEFAULT_PRESET])
    clamped_notes = []
    for src, dst in keys.items():
        if src not in preset:
            continue
        raw = _number(preset[src], src)
        if dst == "top_k" and raw <= 0:
            raw = 0  # SillyTavern disables top-k with -1, llama.cpp with 0.
        lo, hi = store.PARAM_RANGES[dst]
        clamped = min(max(raw, lo), hi)
        if clamped != raw:
            clamped_notes.append(f"{src} {raw:g} → {clamped:g}")
        # The default's type says whether the key holds an int or a float.
        values[dst] = type(values[dst])(clamped)

    for key, (label, neutral) in _NEUTRAL.items():
        if key in preset and not _is_neutral(preset[key], neutral) \
                and label not in skipped:
            skipped.append(label)

    name = next((n.strip() for n in (preset.get("name"), data.get("name"),
                                      fallback_name)
                 if isinstance(n, str) and n.strip()), "Imported")
    return ImportedPreset(name=name, values=values, skipped=skipped,
                          clamped=clamped_notes)


def parse_text(text: str | bytes, fallback_name: str = "Imported") -> ImportedPreset:
    if len(text) > MAX_BYTES:
        raise PresetFormatError("That file is too large to be a preset.")
    try:
        # From bytes, json detects the encoding and skips a BOM.
        data = json.loads(text)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise PresetFormatError(f"Not valid JSON: {exc}") from exc
    return parse_preset(data, fallback_name)


def _raw_url(url: str) -> str:
    """A Hugging Face or GitHub page link -> the file's raw download link."""
    url = re.sub(r"^(https://huggingface\.co/.+?)/blob/", r"\1/resolve/", url)
    return re.sub(r"^https://github\.com/([^/]+/[^/]+)/blob/",
                  r"https://raw.githubusercontent.com/\1/", url)


def fetch(url: str) -> ImportedPreset:
    url = _raw_url(url.strip())
    if not url.startswith(("https://", "http://")):
        raise PresetFormatError("Enter an http(s) link to a .json file.")
    # timeout bounds each read; the deadline bounds a server that trickles.
    deadline = time.monotonic() + FETCH_DEADLINE_S
    with requests.get(url, stream=True, timeout=FETCH_DEADLINE_S) as r:
        if not r.ok:
            raise PresetFormatError(f"Download failed: HTTP {r.status_code}")
        body = b""
        for chunk in r.iter_content(64 * 1024):
            body += chunk
            if len(body) > MAX_BYTES:
                raise PresetFormatError("That file is too large to be a preset.")
            if time.monotonic() > deadline:
                raise PresetFormatError("Download timed out.")
    return parse_text(body, stem_name(url))
