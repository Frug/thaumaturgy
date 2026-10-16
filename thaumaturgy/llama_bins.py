"""Resolve llama.cpp command-line binaries.

llama_install unpacks upstream's prebuilt release into <data>/llama.cpp/bin.
$THAUM_LLAMA_SERVER points at a llama-server built or installed elsewhere
instead; its sibling tools (llama-quantize) are looked up next to it.
"""

import os
import sys
from pathlib import Path

from thaumaturgy.paths import sub_dir


def exe(name: str) -> str:
    return f"{name}.exe" if sys.platform == "win32" else name


def local_bin_dir() -> Path:
    return sub_dir("llama.cpp") / "bin"


def overridden() -> bool:
    return bool(os.environ.get("THAUM_LLAMA_SERVER"))


def server_path() -> Path:
    override = os.environ.get("THAUM_LLAMA_SERVER")
    if override:
        return Path(override)
    return local_bin_dir() / exe("llama-server")


def installed() -> bool:
    return server_path().exists()


def binary_path(name: str) -> Path:
    return server_path().parent / exe(name)
