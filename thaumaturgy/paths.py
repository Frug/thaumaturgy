"""Portable data directory resolution.

Everything user-owned (chats, characters, presets, config) lives under a single
data dir so it can be tarred up and moved. Order of precedence:
  1. $THAUM_DATA
  2. ./data  (relative to the working directory)
"""

import os
from pathlib import Path


def data_dir() -> Path:
    base = Path(os.environ.get("THAUM_DATA") or (Path.cwd() / "data"))
    base.mkdir(parents=True, exist_ok=True)
    return base


def sub_dir(name: str) -> Path:
    p = data_dir() / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def chats_dir() -> Path:
    return sub_dir("chats")
