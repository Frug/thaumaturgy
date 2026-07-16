"""Install the latest upstream llama.cpp release into data/llama.cpp/bin.

Usage:
    uv run python scripts/install_latest_llama_cpp.py cpu
    uv run python scripts/install_latest_llama_cpp.py vulkan
    uv run python scripts/install_latest_llama_cpp.py rocm-7.2

The app will prefer data/llama.cpp/bin/llama-server over the bundled wheel.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path


API_URL = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
ASSET_BY_BACKEND = {
    "cpu": "bin-ubuntu-x64.tar.gz",
    "vulkan": "bin-ubuntu-vulkan-x64.tar.gz",
    "rocm-7.2": "bin-ubuntu-rocm-7.2-x64.tar.gz",
    "openvino": "bin-ubuntu-openvino-2026.2.1-x64.tar.gz",
    "sycl-fp16": "bin-ubuntu-sycl-fp16-x64.tar.gz",
    "sycl-fp32": "bin-ubuntu-sycl-fp32-x64.tar.gz",
}


def _read_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _download(url: str, path: Path) -> None:
    with urllib.request.urlopen(url, timeout=600) as response:
        with path.open("wb") as f:
            shutil.copyfileobj(response, f)


def _asset(release: dict, backend: str) -> dict:
    suffix = ASSET_BY_BACKEND[backend]
    for asset in release.get("assets", []):
        if asset.get("name", "").endswith(suffix):
            return asset
    raise RuntimeError(f"No Linux x64 {backend} asset found in {release.get('tag_name')}.")


def _copy_bins(extracted: Path, dest: Path) -> None:
    candidates = list(extracted.rglob("llama-server"))
    if not candidates:
        raise RuntimeError("The release archive did not contain llama-server.")
    source_bin = candidates[0].parent
    dest.mkdir(parents=True, exist_ok=True)
    for binary in source_bin.iterdir():
        if binary.is_file() and binary.name.startswith("llama"):
            target = dest / binary.name
            shutil.copy2(binary, target)
            target.chmod(target.stat().st_mode | 0o111)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=sorted(ASSET_BY_BACKEND))
    parser.add_argument("--dest", default="data/llama.cpp/bin")
    args = parser.parse_args()

    release = _read_json(API_URL)
    asset = _asset(release, args.backend)
    dest = Path(args.dest)
    print(f"Installing llama.cpp {release['tag_name']} ({args.backend})")
    print(asset["browser_download_url"])

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        archive = tmpdir / asset["name"]
        _download(asset["browser_download_url"], archive)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(tmpdir / "extract", filter="data")
        _copy_bins(tmpdir / "extract", dest)

    print(f"Installed llama.cpp binaries into {dest}")


if __name__ == "__main__":
    main()
