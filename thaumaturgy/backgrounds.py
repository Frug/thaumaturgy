"""Validation and storage for per-chat background images."""

import hashlib
import os
import re
from pathlib import Path

from thaumaturgy.paths import sub_dir

MAX_BYTES = 15 * 1024 * 1024
ACCEPT = ".png,.jpg,.jpeg,.webp,.gif"

_NAME_RE = re.compile(r"[0-9a-f]{16}-[0-9a-f]{16}\.(?:png|jpg|webp|gif)")


def directory() -> Path:
    return sub_dir("chat-backgrounds")


def _extension(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def valid_name(filename: str | None) -> bool:
    return isinstance(filename, str) and _NAME_RE.fullmatch(filename) is not None


def save(chat_id: str, data: bytes) -> str:
    """Validate and atomically save an image, returning its safe filename."""
    if not data:
        raise ValueError("The image is empty.")
    if len(data) > MAX_BYTES:
        raise ValueError("The image must be 15 MB or smaller.")
    extension = _extension(data)
    if extension is None:
        raise ValueError("Choose a PNG, JPEG, WebP, or GIF image.")

    chat_key = hashlib.sha256(chat_id.encode("utf-8")).hexdigest()[:16]
    image_key = hashlib.sha256(data).hexdigest()[:16]
    filename = f"{chat_key}-{image_key}.{extension}"
    target = directory() / filename
    if not target.exists():
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_bytes(data)
        os.replace(temporary, target)
    return filename


def remove(filename: str | None) -> None:
    """Remove one managed image; ignore missing or untrusted filenames."""
    if valid_name(filename):
        (directory() / filename).unlink(missing_ok=True)


def url(filename: str | None) -> str | None:
    """Return the public URL for an existing managed image."""
    if not valid_name(filename) or not (directory() / filename).is_file():
        return None
    return f"/chat-backgrounds/{filename}"
