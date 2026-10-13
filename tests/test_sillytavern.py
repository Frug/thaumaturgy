import json

import pytest

from thaumaturgy import sillytavern, store

# Trimmed from Darkhn/Sampler_settings_and_system_prompt's Gemma 4 master export.
MASTER_EXPORT = {
    "instruct": {"input_sequence": "<|turn>user\n", "name": "Gemma_4_Think"},
    "context": {"story_string": "{{system}}", "name": "Gemma4"},
    "sysprompt": {"name": "Roleplay", "content": "You are {{char}}."},
    "preset": {
        "temp": 1, "top_p": 1, "top_k": -1, "min_p": 0.02, "rep_pen": 1,
        "typical_p": 1, "tfs": 1, "freq_pen": 0, "dry_multiplier": 0,
        "dynatemp": False, "mirostat_mode": 0,
        "logit_bias": [{"id": "x", "text": "68754", "value": -100}],
        "genamt": 2000, "max_length": 56064, "name": "Gemma4",
    },
    "reasoning": {"prefix": "<|channel>thought\n", "name": "gemma4"},
}


def test_master_export_maps_samplers_and_reports_the_rest():
    imported = sillytavern.parse_preset(MASTER_EXPORT)

    assert imported.name == "Gemma4"
    assert imported.values == {
        "max_new_tokens": 2000, "temperature": 1.0, "top_p": 1.0, "top_k": 0,
        "min_p": 0.02, "repetition_penalty": 1.0,
        "recap_tokens": store.BUILTIN_PRESETS[store.DEFAULT_PRESET]["recap_tokens"],
    }
    assert imported.skipped == ["instruct template", "context template",
                                "system prompt", "reasoning format", "logit bias"]


def test_bare_textgen_preset_uses_defaults_for_missing_keys():
    imported = sillytavern.parse_text(
        json.dumps({"temp": 0.7, "dry_multiplier": 0.8}), "My Preset")

    default = store.BUILTIN_PRESETS[store.DEFAULT_PRESET]
    assert imported.name == "My Preset"
    assert imported.values["temperature"] == 0.7
    assert imported.values["top_p"] == default["top_p"]
    assert imported.skipped == ["DRY"]


def test_chat_completion_preset():
    imported = sillytavern.parse_preset({
        "temperature": 0.9, "top_k": 50, "repetition_penalty": 1.1,
        "openai_max_tokens": 800, "presence_penalty": 0.3,
    })

    assert imported.values["temperature"] == 0.9
    assert imported.values["top_k"] == 50
    assert imported.values["max_new_tokens"] == 800
    assert imported.skipped == ["presence penalty"]


@pytest.mark.parametrize("text", [
    "not json", "[]", json.dumps({"name": "x"}), json.dumps({"temp": "hot"}),
    '{"temp": NaN}', '{"temp": Infinity}', '{"top_p": 1e400}',
    '{"temp": 1, "top_k": 1' + "0" * 400 + '}', b'{"temp": "\xff"}',
])
def test_rejects_non_presets(text):
    with pytest.raises(sillytavern.PresetFormatError):
        sillytavern.parse_text(text)


def test_page_links_become_raw_links():
    assert sillytavern._raw_url(
        "https://huggingface.co/a/b/blob/main/p.json") \
        == "https://huggingface.co/a/b/resolve/main/p.json"
    assert sillytavern._raw_url("https://github.com/a/b/blob/main/p.json") \
        == "https://raw.githubusercontent.com/a/b/main/p.json"



def test_out_of_range_values_are_clamped_and_reported():
    imported = sillytavern.parse_preset({
        "temp": 5, "top_p": 3, "min_p": -1, "rep_pen": 0.9, "genamt": 8192,
        "top_k": 99999999999999999999,
    })

    assert imported.values["temperature"] == 2.0
    assert imported.values["top_p"] == 1.0
    assert imported.values["min_p"] == 0.0
    assert imported.values["repetition_penalty"] == 1.0
    assert imported.values["max_new_tokens"] == 4096
    assert imported.values["top_k"] == 200
    assert "genamt 8192 → 4096" in imported.clamped
    assert len(imported.clamped) == 6
    assert imported.skipped == []


def test_bom_and_empty_containers():
    imported = sillytavern.parse_text(
        b"\xef\xbb\xbf" + json.dumps({"temp": 1, "logit_bias": {}}).encode())

    assert imported.values["temperature"] == 1.0
    assert imported.skipped == []


@pytest.mark.parametrize("path,stem", [
    ("https://hf.co/x/resolve/main/Universal-Light.json?download=true",
     "Universal-Light"),
    ("https://hf.co/x/Big%20O.json", "Big O"),
    ("Preset.JSON", "Preset"),
])
def test_stem_name(path, stem):
    assert sillytavern.stem_name(path) == stem


@pytest.mark.parametrize("data,fallback", [
    ({"temp": 1}, "  "), ({"temp": 1, "name": {"a": 1}}, ""),
])
def test_blank_or_odd_names_fall_back(data, fallback):
    assert sillytavern.parse_preset(data, fallback).name == "Imported"
