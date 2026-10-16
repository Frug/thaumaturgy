import hashlib
import io
import tarfile
import zipfile

import pytest

from thaumaturgy import llama_bins, llama_install
from thaumaturgy.llama_install import Host

# Abridged from the b11680 release.
NAMES = [
    "cudart-llama-b11680-bin-ubuntu-cuda-12.8-x64.tar.gz",
    "cudart-llama-b11680-bin-ubuntu-cuda-13.4-x64.tar.gz",
    "cudart-llama-bin-win-cuda-12.4-x64.zip",
    "cudart-llama-bin-win-cuda-13.4-x64.zip",
    "llama-b11680-bin-macos-arm64.tar.gz",
    "llama-b11680-bin-ubuntu-arm64.tar.gz",
    "llama-b11680-bin-ubuntu-cuda-12.8-x64.tar.gz",
    "llama-b11680-bin-ubuntu-cuda-13.4-x64.tar.gz",
    "llama-b11680-bin-ubuntu-rocm-10.0-x64.tar.gz",
    "llama-b11680-bin-ubuntu-vulkan-arm64.tar.gz",
    "llama-b11680-bin-ubuntu-vulkan-x64.tar.gz",
    "llama-b11680-bin-ubuntu-x64.tar.gz",
    "llama-b11680-bin-win-cpu-x64.zip",
    "llama-b11680-bin-win-cuda-12.4-x64.zip",
    "llama-b11680-bin-win-cuda-13.4-x64.zip",
    "llama-b11680-bin-win-vulkan-x64.zip",
    "llama-b11680-ui.tar.gz",
]
ASSETS = [{"name": n} for n in NAMES]


def picked(host: Host) -> tuple[str, list[str]]:
    variant, assets = llama_install.pick_assets(ASSETS, host)
    return variant, [a["name"] for a in assets]


@pytest.mark.parametrize("host, expected", [
    (Host("ubuntu", "x64", (13, 2)), ("cuda-13.4", [
        "llama-b11680-bin-ubuntu-cuda-13.4-x64.tar.gz",
        "cudart-llama-b11680-bin-ubuntu-cuda-13.4-x64.tar.gz"])),
    (Host("ubuntu", "x64", (12, 4)), ("cuda-12.8", [
        "llama-b11680-bin-ubuntu-cuda-12.8-x64.tar.gz",
        "cudart-llama-b11680-bin-ubuntu-cuda-12.8-x64.tar.gz"])),
    (Host("win", "x64", (13, 0)), ("cuda-13.4", [
        "llama-b11680-bin-win-cuda-13.4-x64.zip",
        "cudart-llama-bin-win-cuda-13.4-x64.zip"])),
    # A driver too old for any CUDA build still has a GPU Vulkan can use.
    (Host("ubuntu", "x64", (11, 8)),
     ("vulkan", ["llama-b11680-bin-ubuntu-vulkan-x64.tar.gz"])),
    (Host("ubuntu", "x64", None),
     ("vulkan", ["llama-b11680-bin-ubuntu-vulkan-x64.tar.gz"])),
    (Host("win", "x64", None), ("vulkan", ["llama-b11680-bin-win-vulkan-x64.zip"])),
    (Host("macos", "arm64", None), ("metal", ["llama-b11680-bin-macos-arm64.tar.gz"])),
    # No arm64 CUDA build in this list, so the driver doesn't matter.
    (Host("ubuntu", "arm64", (13, 2)),
     ("vulkan", ["llama-b11680-bin-ubuntu-vulkan-arm64.tar.gz"])),
])
def test_pick_assets(host, expected):
    assert picked(host) == expected


def test_pick_assets_falls_back_to_cpu_without_vulkan():
    cpu_only = [a for a in ASSETS if "vulkan" not in a["name"]]
    variant, assets = llama_install.pick_assets(cpu_only, Host("win", "x64", None))
    assert (variant, assets[0]["name"]) == ("cpu", "llama-b11680-bin-win-cpu-x64.zip")


def test_pick_assets_rejects_unsupported_host():
    with pytest.raises(RuntimeError):
        llama_install.pick_assets(ASSETS, Host("macos", "x64", None))


def test_driver_cuda():
    header = "| NVIDIA-SMI 595.91.07   Driver Version: 595.91.07   CUDA Version: 13.2 |"
    assert llama_install.driver_cuda(header) == (13, 2)
    assert llama_install.driver_cuda("command not found") is None


def _tarball(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o755
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _release(archives: dict[str, bytes]) -> dict:
    return {"tag_name": "b2", "assets": [
        {"name": name, "size": len(data), "browser_download_url": name,
         "digest": "sha256:" + hashlib.sha256(data).hexdigest()}
        for name, data in archives.items()]}


@pytest.fixture
def fake_downloads(monkeypatch, tmp_path):
    """Serve archives by name instead of from GitHub."""
    monkeypatch.setenv("THAUM_DATA", str(tmp_path / "data"))
    monkeypatch.delenv("THAUM_LLAMA_SERVER", raising=False)
    served: dict[str, bytes] = {}

    class Response:
        def __init__(self, url):
            self.data = served[url]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield self.data

    monkeypatch.setattr(llama_install.requests, "get",
                        lambda url, **_kw: Response(url))
    return served


def test_install_unpacks_tarballs_and_runtime_into_bin(fake_downloads):
    archives = {
        "llama-b2-bin-ubuntu-cuda-13.4-x64.tar.gz": _tarball({
            "llama-b2/llama-server": b"server", "llama-b2/libggml-cuda.so": b"so"}),
        "cudart-llama-b2-bin-ubuntu-cuda-13.4-x64.tar.gz": _tarball({
            "cudart-llama-b2-bin-ubuntu-cuda-13.4-x64/libcudart.so.13": b"rt"}),
    }
    fake_downloads.update(archives)
    bin_dir = llama_bins.local_bin_dir()
    bin_dir.mkdir()
    (bin_dir / "stale-from-last-build.so").write_bytes(b"old")

    done = llama_install.install(release=_release(archives),
                                 host=Host("ubuntu", "x64", (13, 2)))

    assert done == "b2 (cuda-13.4)"
    assert sorted(p.name for p in bin_dir.iterdir()) == [
        "libcudart.so.13", "libggml-cuda.so", "llama-server"]
    assert llama_install.installed_backend() == "CUDA"
    assert sorted(p.name for p in bin_dir.parent.iterdir()) == ["bin"]


def test_install_unpacks_flat_zip(fake_downloads):
    archives = {"llama-b2-bin-win-vulkan-x64.zip": _zip({
        llama_bins.exe("llama-server"): b"server", "ggml-vulkan.dll": b"dll"})}
    fake_downloads.update(archives)

    llama_install.install(release=_release(archives), host=Host("win", "x64", None))

    assert llama_bins.installed()
    assert llama_install.installed_backend() == "Vulkan"


def test_failed_install_keeps_the_working_one(fake_downloads):
    archives = {"llama-b2-bin-ubuntu-vulkan-x64.tar.gz": _tarball({
        "llama-b2/llama-server": b"server"})}
    fake_downloads.update(archives)
    release = _release(archives)
    release["assets"][0]["digest"] = "sha256:" + "0" * 64
    working = llama_bins.server_path()
    working.parent.mkdir()
    working.write_bytes(b"current")

    with pytest.raises(RuntimeError, match="Checksum"):
        llama_install.install(release=release, host=Host("ubuntu", "x64", None))

    assert working.read_bytes() == b"current"
    assert sorted(p.name for p in working.parent.parent.iterdir()) == ["bin"]


def test_override_points_tools_next_to_it(monkeypatch, tmp_path):
    monkeypatch.setenv("THAUM_LLAMA_SERVER", str(tmp_path / "build" / "llama-server"))
    assert llama_bins.overridden()
    assert llama_bins.binary_path("llama-quantize") == \
        tmp_path / "build" / llama_bins.exe("llama-quantize")
