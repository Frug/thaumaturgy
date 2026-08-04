"""Portable data directory resolution.

Everything user-owned (chats, scenarios, presets, config) lives under a single
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


def log_dir() -> Path | None:
    """Where to write diagnostic logs, or None when logging is off.

    Opt-in via $THAUM_LOG_DIR. llama-server's output is otherwise held in a
    500-line ring buffer in memory, so the per-request timings and the
    layer-offload counts it prints are gone by the time anyone asks about them.
    """
    raw = os.environ.get("THAUM_LOG_DIR")
    if not raw:
        return None
    path = Path(raw).expanduser()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return path
