"""Download models from Hugging Face into the models dir.

Two paths, chosen by what the repo actually contains:

  * Repo already has GGUFs → download the file(s) matching the requested quant.
    Light path: only needs huggingface_hub (a core dependency).

  * Repo is safetensors → snapshot it, convert to an f16 GGUF with llama.cpp's
    convert_hf_to_gguf.py (auto-fetched, pinned to the bundled binary's build
    commit), then quantize with the bundled llama-quantize. Heavy path: needs the
    convert deps (torch/transformers/gguf/sentencepiece) from the `training` extra.

All long steps take an on_progress(str) callback so the UI can show stage text.
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import requests

import llama_cpp_binaries

from thaumaturgy.engine import models_dir
from thaumaturgy.paths import sub_dir

CONVERTER_URL = ("https://raw.githubusercontent.com/ggml-org/llama.cpp/"
                 "{commit}/convert_hf_to_gguf.py")


def parse_repo_id(url: str) -> str:
    """Extract 'owner/model' from a HF URL (or accept a bare 'owner/model')."""
    s = (url or "").strip()
    s = re.sub(r"^(https?://)?(www\.)?huggingface\.co/", "", s)
    s = s.split("?")[0].split("#")[0]
    # Drop /tree/<rev>, /blob/<rev>, /resolve/<rev>, etc.
    s = re.split(r"/(?:tree|blob|resolve)/", s)[0]
    s = s.strip("/")
    parts = s.split("/")
    if len(parts) < 2:
        raise ValueError(f"Could not parse a HF repo id from: {url!r}")
    return "/".join(parts[:2])


def _bin_dir() -> Path:
    return Path(llama_cpp_binaries.get_binary_path()).parent


def _build_commit() -> str:
    out = subprocess.run([str(_bin_dir() / "llama-server"), "--version"],
                         capture_output=True, text=True, timeout=30)
    m = re.search(r"\(([0-9a-f]{7,40})\)", (out.stdout or "") + (out.stderr or ""))
    if not m:
        raise RuntimeError("Could not read llama.cpp build commit from the binary.")
    return m.group(1)


def _converter_path(on_progress) -> Path:
    commit = _build_commit()
    cache = sub_dir("cache") / "convert"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"convert_hf_to_gguf-{commit}.py"
    if not path.exists():
        on_progress(f"Fetching converter (llama.cpp @ {commit})…")
        r = requests.get(CONVERTER_URL.format(commit=commit), timeout=60)
        r.raise_for_status()
        path.write_text(r.text, encoding="utf-8")
    return path


def _missing_convert_deps() -> list[str]:
    missing = []
    for mod in ("torch", "gguf", "sentencepiece"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    return missing


def _pick_gguf_files(ggufs: list[str], quant: str) -> list[str]:
    """Files whose name carries the requested quant token (handles shards)."""
    token = quant.lower()
    matches = [f for f in ggufs if token in f.lower()]
    if not matches:
        avail = sorted({m.group(0) for f in ggufs
                        if (m := re.search(r"(?i)q\d[_a-z0-9]*", f))})
        hint = f" Available: {', '.join(avail)}." if avail else ""
        raise ValueError(f"No {quant} GGUF in this repo.{hint}")
    return matches


def _run_streamed(cmd: list[str], on_progress, label: str) -> None:
    on_progress(f"{label}…")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    tail = ""
    for line in proc.stdout:
        tail = line.strip() or tail
        if tail:
            on_progress(f"{label}: {tail[:80]}")
    if proc.wait() != 0:
        raise RuntimeError(f"{label} failed (exit {proc.returncode}): {tail[:200]}")


def download(url: str, quant: str, on_progress=lambda _m: None) -> str:
    """Download (converting/quantizing if needed) and return the GGUF filename."""
    from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

    repo_id = parse_repo_id(url)
    model_name = repo_id.split("/")[1]
    on_progress(f"Resolving {repo_id}…")
    files = list_repo_files(repo_id)
    ggufs = [f for f in files if f.lower().endswith(".gguf")]

    if ggufs:
        targets = _pick_gguf_files(ggufs, quant)
        last = None
        for i, rel in enumerate(targets, 1):
            on_progress(f"Downloading GGUF {i}/{len(targets)}: {os.path.basename(rel)}…")
            local = hf_hub_download(repo_id, rel)
            dest = models_dir() / os.path.basename(rel)
            shutil.copyfile(local, dest)
            last = dest.name
        return last

    # ── safetensors → convert → quantize ─────────────────────────────────────
    missing = _missing_convert_deps()
    if missing:
        raise RuntimeError(
            "This repo has no GGUF, so it needs conversion. Missing "
            f"{', '.join(missing)} — run `uv sync --extra training` and retry.")

    converter = _converter_path(on_progress)
    work = sub_dir("cache") / "convert" / model_name
    on_progress(f"Downloading {repo_id} (safetensors)…")
    snapshot_download(
        repo_id, local_dir=str(work),
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model",
                        "tokenizer*", "*.tiktoken", "*.jinja"])

    f16 = work / f"{model_name}-f16.gguf"
    _run_streamed([sys.executable, str(converter), str(work),
                   "--outfile", str(f16), "--outtype", "f16"],
                  on_progress, "Converting to GGUF")

    out = models_dir() / f"{model_name}-{quant}.gguf"
    _run_streamed([str(_bin_dir() / "llama-quantize"), str(f16), str(out), quant],
                  on_progress, f"Quantizing to {quant}")

    on_progress("Cleaning up…")
    shutil.rmtree(work, ignore_errors=True)
    return out.name
