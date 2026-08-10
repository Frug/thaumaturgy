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


_log_dir_raw: str | None = None


def reset_log_dir() -> None:
    """Forget the cached setting after Settings writes a new one."""
    global _log_dir_raw
    _log_dir_raw = None


def _log_dir_raw_value() -> str:
    # Cached: this is consulted once per line of llama-server output, and the
    # setting lives in a YAML file.
    global _log_dir_raw
    if _log_dir_raw is None:
        from thaumaturgy import store
        _log_dir_raw = (os.environ.get("THAUM_LOG_DIR") or "").strip() \
            or store.log_dir_setting()
    return _log_dir_raw


def log_dir() -> Path | None:
    """Where to write diagnostic logs, or None when logging is off.

    Opt-in on the Settings page, or via $THAUM_LOG_DIR, which wins over it.
    llama-server's output is otherwise held in a 500-line ring buffer in
    memory, so the per-request timings and the layer-offload counts it prints
    are gone by the time anyone asks about them.
    """
    raw = _log_dir_raw_value()
    if not raw:
        return None
    path = Path(raw).expanduser()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return path
