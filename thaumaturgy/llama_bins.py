"""Resolve llama.cpp command-line binaries.

The Python wheel gives us a convenient default, but upstream llama.cpp releases
ship as binary tarballs. Prefer an explicit/local install when present so the
runtime can be updated independently of the wheel.
"""

import os
from pathlib import Path

import llama_cpp_binaries

from thaumaturgy.paths import sub_dir


def _local_bin_dir() -> Path:
    return sub_dir("llama.cpp") / "bin"


def server_path() -> Path:
    override = os.environ.get("THAUM_LLAMA_SERVER")
    if override:
        return Path(override)

    local = _local_bin_dir() / "llama-server"
    if local.exists():
        return local

    return Path(llama_cpp_binaries.get_binary_path())


def bin_dir() -> Path:
    return server_path().parent


def binary_path(name: str) -> Path:
    path = bin_dir() / name
    if path.exists():
        return path

    fallback = Path(llama_cpp_binaries.get_binary_path()).parent / name
    return fallback
