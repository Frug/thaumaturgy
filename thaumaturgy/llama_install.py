"""Install and update llama.cpp from upstream's prebuilt releases.

ggml-org publishes a release for every build (tagged bNNNNN) with archives per
platform and GPU backend. We pick the one that suits this machine, unpack it
into <data>/llama.cpp/bin, and llama_bins resolves the server from there.

The pick, in order: NVIDIA's CUDA build when the driver can run one, then
Vulkan, which covers AMD and Intel GPUs too and runs on the CPU when no device
is found, then the plain CPU build. macOS has a single build, with Metal.
"""

import hashlib
import platform
import re
import shutil
import subprocess
import tarfile
import threading
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests

from thaumaturgy import llama_bins

RELEASES_URL = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10"
_TAG_RE = re.compile(r"^b\d+$")
# e.g. "llama-b11680-bin-ubuntu-cuda-13.4-x64.tar.gz"; the variant is absent
# from the plain Linux CPU build and from the macOS one.
_ASSET_RE = re.compile(
    r"^llama-b\d+-bin-(ubuntu|win|macos)-(?:(.+)-)?(x64|arm64)\.(?:tar\.gz|zip)$")
_CUDA_RE = re.compile(r"^cuda-(\d+)\.(\d+)$")
_DRIVER_CUDA_RE = re.compile(r"CUDA Version:\s*(\d+)\.(\d+)")
_OS_NAMES = {"Linux": "ubuntu", "Windows": "win", "Darwin": "macos"}
_ARCH_NAMES = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64"}
# Held for a whole install, so a load can't start from a half-swapped tree and
# two pages can't install at once.
_lock = threading.Lock()


def busy() -> bool:
    return _lock.locked()


@dataclass(frozen=True)
class Host:
    os: str  # as upstream names it in asset names: ubuntu, win, macos
    arch: str  # x64 or arm64
    cuda: tuple[int, int] | None  # the newest CUDA the NVIDIA driver supports


def driver_cuda(nvidia_smi_output: str) -> tuple[int, int] | None:
    m = _DRIVER_CUDA_RE.search(nvidia_smi_output)
    return (int(m.group(1)), int(m.group(2))) if m else None


def detect_host() -> Host:
    os_name = _OS_NAMES.get(platform.system())
    arch = _ARCH_NAMES.get(platform.machine().lower())
    if not os_name or not arch:
        raise RuntimeError(
            f"No prebuilt llama.cpp for {platform.system()} {platform.machine()}. "
            "Build it yourself and point THAUM_LLAMA_SERVER at llama-server.")
    cuda = None
    if os_name != "macos":
        try:
            out = subprocess.run(["nvidia-smi"], capture_output=True, text=True,
                                 timeout=15)
            cuda = driver_cuda(out.stdout)
        except (OSError, subprocess.SubprocessError):
            pass
    return Host(os_name, arch, cuda)


def latest_release() -> dict:
    """The newest build release. The repo's vX.Y.Z releases carry no binaries."""
    r = requests.get(RELEASES_URL, timeout=30,
                     headers={"Accept": "application/vnd.github+json"})
    r.raise_for_status()
    for release in r.json():
        if _TAG_RE.match(release.get("tag_name", "")) and release.get("assets"):
            return release
    raise RuntimeError("No llama.cpp build release found on GitHub.")


def pick_assets(assets: list[dict], host: Host) -> tuple[str, list[dict]]:
    """Choose the build for `host`: (variant label, archives to unpack in order)."""
    builds: dict[str | None, dict] = {}
    for asset in assets:
        m = _ASSET_RE.match(asset["name"])
        if m and m.group(1) == host.os and m.group(3) == host.arch:
            builds[m.group(2)] = asset

    if host.cuda:
        # CUDA's minor-version compatibility lets a 13.4 build run on a 13.2
        # driver, so only the major version has to match.
        usable = []
        for variant in builds:
            m = _CUDA_RE.match(variant or "")
            if m and int(m.group(1)) <= host.cuda[0]:
                usable.append(((int(m.group(1)), int(m.group(2))), variant))
        if usable:
            variant = max(usable)[1]
            # The CUDA runtime ships separately; Windows' archive is unversioned.
            runtime = [a for a in assets if a["name"].startswith("cudart-")
                       and f"-bin-{host.os}-{variant}-{host.arch}." in a["name"]]
            return variant, [builds[variant], *runtime]

    for variant in ("vulkan", None, "cpu"):
        if variant in builds:
            label = variant or ("metal" if host.os == "macos" else "cpu")
            return label, [builds[variant]]
    raise RuntimeError(f"This release has no llama.cpp build for {host.os} {host.arch}.")


def installed_build() -> str | None:
    """The build tag of the resolved llama-server, e.g. "b11680"."""
    path = llama_bins.server_path()
    if not path.exists():
        return None
    try:
        out = subprocess.run([str(path), "--version"], capture_output=True,
                             text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"build (\d+)", (out.stdout or "") + (out.stderr or ""))
    return f"b{m.group(1)}" if m else None


def installed_backend() -> str | None:
    """The GPU backend in the managed install, judged by which library it has."""
    bin_dir = llama_bins.local_bin_dir()
    names = {p.name for p in bin_dir.iterdir()} if bin_dir.exists() else set()
    for backend, label in (("cuda", "CUDA"), ("vulkan", "Vulkan"), ("metal", "Metal")):
        if any(f"ggml-{backend}" in name for name in names):
            return label
    return "CPU" if names else None


def _download(asset: dict, dest: Path, on_progress) -> Path:
    path = dest / asset["name"]
    total = asset.get("size") or 0
    digest = hashlib.sha256()
    done = 0
    with requests.get(asset["browser_download_url"], stream=True, timeout=60) as r:
        r.raise_for_status()
        with path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                on_progress(f"Downloading {asset['name']}: "
                            f"{done // 1_000_000} / {total // 1_000_000} MB")
    # GitHub reports a digest for assets uploaded since mid-2025.
    expected = (asset.get("digest") or "").removeprefix("sha256:")
    if expected and digest.hexdigest() != expected:
        raise RuntimeError(f"Checksum mismatch for {asset['name']}.")
    return path


def _extract(archive: Path, dest: Path) -> None:
    """Unpack into `dest`, dropping the single top-level folder tarballs have."""
    unpacked = dest.with_name(dest.name + ".unpack")
    shutil.rmtree(unpacked, ignore_errors=True)
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(unpacked)
    else:
        with tarfile.open(archive) as tf:
            tf.extractall(unpacked, filter="data")
    entries = list(unpacked.iterdir())
    root = entries[0] if len(entries) == 1 and entries[0].is_dir() else unpacked
    dest.mkdir(parents=True, exist_ok=True)
    for item in root.iterdir():
        item.replace(dest / item.name)
    shutil.rmtree(unpacked, ignore_errors=True)


def install(on_progress=lambda _: None, release: dict | None = None,
            host: Host | None = None) -> str:
    """Install the latest (or the given) release, replacing any managed install.

    The caller must make sure llama-server isn't running from it: Windows
    won't let the files be replaced, and elsewhere the old process would carry
    on from deleted files. Returns a description of what was installed.
    """
    if not _lock.acquire(blocking=False):
        raise RuntimeError("llama.cpp is already being installed.")
    try:
        return _install(on_progress, release, host)
    finally:
        _lock.release()


def _install(on_progress, release: dict | None, host: Host | None) -> str:
    on_progress("Checking for the latest llama.cpp…")
    release = release or latest_release()
    host = host or detect_host()
    variant, assets = pick_assets(release["assets"], host)

    target = llama_bins.local_bin_dir()
    work = target.parent / "download"
    staging = target.with_name("bin.partial")
    shutil.rmtree(work, ignore_errors=True)
    shutil.rmtree(staging, ignore_errors=True)
    work.mkdir(parents=True)
    try:
        for asset in assets:
            archive = _download(asset, work, on_progress)
            on_progress(f"Unpacking {asset['name']}…")
            _extract(archive, staging)
            archive.unlink()
        if not (staging / llama_bins.exe("llama-server")).exists():
            raise RuntimeError(f"{assets[0]['name']} has no llama-server.")
        # Swap only once the new tree is complete, so a failed download leaves
        # the working install alone.
        shutil.rmtree(target, ignore_errors=True)
        staging.replace(target)
    finally:
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(staging, ignore_errors=True)
    return f"{release['tag_name']} ({variant})"
