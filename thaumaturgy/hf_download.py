"""Download models from Hugging Face into the models dir.

Two paths, chosen by what the repo actually contains:

  * Repo already has GGUFs → download the file(s) matching the requested quant.
    Light path: only needs huggingface_hub (a core dependency).

  * Repo is safetensors → snapshot it, convert to an f16 GGUF with llama.cpp's
    convert_hf_to_gguf.py (auto-fetched, pinned to the bundled binary's build
    commit), then quantize with the resolved llama-quantize. Heavy path: needs the
    convert deps (torch/transformers/gguf/sentencepiece) from the `training` extra.

All long steps take an on_progress(str) callback so the UI can show stage text.
"""

import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import requests

from thaumaturgy import llama_bins
from thaumaturgy.engine import models_dir
from thaumaturgy.paths import sub_dir

CONVERTER_URL = "https://codeload.github.com/ggml-org/llama.cpp/tar.gz/{commit}"
# convert_hf_to_gguf.py is only a CLI over the repo's `conversion` package, so
# the script alone raises ModuleNotFoundError. Both come from the same tarball,
# which is ~33MB on the wire and ~900KB once filtered down to these.
CONVERTER_MEMBERS = ("convert_hf_to_gguf.py", "conversion/")
# A split model's parts, e.g. "model-Q4_K_M-00001-of-00003.gguf".
_SHARD_RE = re.compile(r"(?i)-\d{5}-of-\d{5}\.gguf$")
# Just the quant itself, stopping before build suffixes like "_hb16" — those
# are already visible in the label, and swallowing them makes the badge lie.
# Longest forms first: Q4_K_M must win over Q4_K.
_QUANT_RE = re.compile(
    r"(?i)(iq\d+_[a-z]+|q\d+_k_[sml]|q\d+_k|q\d+_\d|bf16|f16|f32)")
# llama-quantize's per-tensor counter, e.g. "[  12/ 291] blk.0.attn_q.weight ...".
_TENSOR_COUNT_RE = re.compile(r"\[\s*(\d+)\s*/\s*(\d+)\s*\]")
# The converter logs through logging, so every line arrives as "INFO:hf-to-gguf:...".
_LOG_PREFIX_RE = re.compile(r"^(?:INFO|WARNING|ERROR|DEBUG):[\w.-]+:")


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


def _build_commit() -> str:
    out = subprocess.run([str(llama_bins.server_path()), "--version"],
                         capture_output=True, text=True, timeout=30)
    m = re.search(r"\(([0-9a-f]{7,40})\)", (out.stdout or "") + (out.stderr or ""))
    if not m:
        raise RuntimeError("Could not read llama.cpp build commit from the binary.")
    return m.group(1)


def _extract_converter(commit: str, dest: Path) -> None:
    """Stream the pinned tarball, keeping only the converter and its package."""
    with requests.get(CONVERTER_URL.format(commit=commit),
                      stream=True, timeout=300) as r:
        r.raise_for_status()
        with tarfile.open(fileobj=r.raw, mode="r|gz") as tf:
            for member in tf:
                # Drop the tarball's "llama.cpp-<commit>/" prefix.
                _, _, rel = member.name.partition("/")
                if rel == CONVERTER_MEMBERS[0] or rel.startswith(CONVERTER_MEMBERS[1]):
                    member.name = rel
                    tf.extract(member, dest, filter="data")
    if not (dest / CONVERTER_MEMBERS[0]).exists():
        raise RuntimeError(f"llama.cpp @ {commit} has no {CONVERTER_MEMBERS[0]}")


def _prune_converters(keep: Path) -> None:
    for old in keep.parent.iterdir():
        if old != keep:
            shutil.rmtree(old, ignore_errors=True)
    # Pre-package layout: a bare script that can no longer run on its own.
    for legacy in (sub_dir("cache") / "convert").glob("convert_hf_to_gguf-*.py"):
        legacy.unlink(missing_ok=True)


def _converter_path(on_progress) -> Path:
    """Fetch convert_hf_to_gguf.py plus the `conversion` package it imports.

    Kept next to each other and pinned to the bundled binary's build commit,
    since the CLI and the package are only guaranteed to agree within one tree.
    """
    commit = _build_commit()
    root = sub_dir("cache") / "converter" / commit
    script = root / CONVERTER_MEMBERS[0]
    if script.exists():
        return script
    on_progress(f"Fetching converter (llama.cpp @ {commit})…")
    # Extract to one side and swap, so a failed download can't leave a partial
    # tree that the script.exists() check above would take for a warm cache.
    staging = root.with_name(f"{commit}.partial")
    shutil.rmtree(staging, ignore_errors=True)
    try:
        _extract_converter(commit, staging)
        shutil.rmtree(root, ignore_errors=True)
        staging.replace(root)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    _prune_converters(keep=root)
    return script


def _missing_convert_deps() -> list[str]:
    missing = []
    for mod in ("torch", "gguf", "sentencepiece"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    return missing


def _variant_stem(name: str) -> str:
    """Filename minus any shard suffix, so one split model groups as one entry."""
    if _SHARD_RE.search(name):
        return _SHARD_RE.sub("", name)
    return name[:-len(".gguf")] if name.lower().endswith(".gguf") else name


def human_size(n: int) -> str:
    if not n:
        return "?"
    return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


def probe(url: str) -> dict:
    """List a repo's GGUF variants without downloading anything.

    Shards of one model collapse into a single entry, but distinct builds of
    the same quant stay separate: a repo may carry plain, imatrix (`i1`) and
    embedding-precision (`hb16`) cuts of Q4_K_M that differ in quality at the
    same size, and only the caller can say which was meant.
    """
    from huggingface_hub import HfApi

    repo_id = parse_repo_id(url)
    info = HfApi().model_info(repo_id, files_metadata=True)
    groups: dict[str, dict] = {}
    for sibling in info.siblings or []:
        name = sibling.rfilename
        if not name.lower().endswith(".gguf"):
            continue
        stem = _variant_stem(name)
        # imatrix.gguf rides along in imatrix repos: calibration data baked into
        # the i1 builds, not something loadable.
        if os.path.basename(stem).lower() == "imatrix":
            continue
        group = groups.setdefault(stem, {"files": [], "size": 0})
        group["files"].append(name)
        group["size"] += sibling.size or 0

    variants = []
    for stem, group in sorted(groups.items()):
        quant = _QUANT_RE.search(os.path.basename(stem))
        variants.append({
            "label": os.path.basename(stem),
            "files": sorted(group["files"]),
            "size": group["size"],
            "quant": quant.group(0).upper() if quant else "",
        })
    return {"repo_id": repo_id, "variants": variants}


def fetch_variant(repo_id: str, files: list[str], on_progress=lambda _m: None) -> str:
    """Download one already-GGUF variant; returns the filename to load."""
    from huggingface_hub import hf_hub_download

    for i, rel in enumerate(files, 1):
        on_progress(f"Downloading {i}/{len(files)}: {os.path.basename(rel)}…")
        local = hf_hub_download(repo_id, rel)
        shutil.copyfile(local, models_dir() / os.path.basename(rel))
    # A split model is loaded through its first shard; llama.cpp finds the rest.
    return os.path.basename(files[0])


def _progress_line(label: str, line: str) -> str:
    """Render one line of tool output as UI status text.

    llama-quantize prefixes each tensor with "[  12/ 291]", the only true
    progress either tool reports; the converter only gets its latest line.
    """
    m = _TENSOR_COUNT_RE.match(line)
    if m:
        done, total = int(m.group(1)), int(m.group(2))
        if total:
            return f"{label}: {done * 100 // total}% ({done}/{total} tensors)"
    return f"{label}: {_LOG_PREFIX_RE.sub('', line)[:80]}"


def _run_streamed(cmd: list[str], on_progress, label: str) -> None:
    on_progress(f"{label}…")
    # errors="replace": both tools echo raw vocab bytes that aren't valid UTF-8
    # on their own, and a strict decode kills this reader part-way through.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", bufsize=1)
    tail = ""
    with proc:  # closes the pipe and reaps the child on the way out
        try:
            for line in proc.stdout:
                line = line.strip()
                if line:
                    tail = line
                    on_progress(_progress_line(label, line))
        except BaseException:
            # Leaving the pipe undrained strands the child blocked on a full
            # one forever, still holding its half-written output file.
            proc.kill()
            raise
        # No kill on the normal path: stdout hits EOF before the child has
        # finished flushing tens of GB, and killing there truncates the result.
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed (exit {proc.returncode}): {tail[:200]}")


def convert(url: str, quant: str, on_progress=lambda _m: None) -> str:
    """Snapshot a safetensors repo, convert to f16 GGUF, quantize to `quant`."""
    from huggingface_hub import snapshot_download

    repo_id = parse_repo_id(url)
    model_name = repo_id.split("/")[1]
    on_progress(f"Resolving {repo_id}…")

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
    _run_streamed([str(llama_bins.binary_path("llama-quantize")), str(f16), str(out), quant],
                  on_progress, f"Quantizing to {quant}")

    on_progress("Cleaning up…")
    shutil.rmtree(work, ignore_errors=True)
    return out.name
