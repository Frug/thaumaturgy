"""Minimal GGUF header reader — pull a model's trained context length without
loading it.

We only need a couple of scalar metadata keys (architecture + context length),
so we parse the KV header and *skip* array bodies (the tokenizer vocab is a
100k+ entry string array — decoding it would make this slow). Reading stops as
soon as the context-length key is found.
"""

import struct

# GGUF value type ids (see ggml/gguf spec).
_UINT8, _INT8, _UINT16, _INT16, _UINT32, _INT32, _FLOAT32, _BOOL, \
    _STRING, _ARRAY, _UINT64, _INT64, _FLOAT64 = range(13)

_PACK = {
    _UINT8: "<B", _INT8: "<b", _UINT16: "<H", _INT16: "<h",
    _UINT32: "<I", _INT32: "<i", _FLOAT32: "<f", _UINT64: "<Q",
    _INT64: "<q", _FLOAT64: "<d", _BOOL: "?",
}
_SIZE = {
    _UINT8: 1, _INT8: 1, _UINT16: 2, _INT16: 2, _UINT32: 4, _INT32: 4,
    _FLOAT32: 4, _UINT64: 8, _INT64: 8, _FLOAT64: 8, _BOOL: 1,
}


def _read_str(f) -> bytes:
    (length,) = struct.unpack("<Q", f.read(8))
    return f.read(length)


def _read_scalar(f, value_type):
    if value_type == _STRING:
        return _read_str(f).decode("utf-8", "replace")
    fmt = _PACK[value_type]
    return struct.unpack(fmt, f.read(_SIZE[value_type]))[0]


def _skip_scalar(f, value_type) -> None:
    if value_type == _STRING:
        (length,) = struct.unpack("<Q", f.read(8))
        f.seek(length, 1)
    else:
        f.seek(_SIZE[value_type], 1)


def read_context_length(fname) -> int | None:
    """Return the model's trained context length from GGUF metadata, or None."""
    with open(fname, "rb") as f:
        if f.read(4) != b"GGUF":
            return None
        (version,) = struct.unpack("<I", f.read(4))
        if version == 1:
            return None
        struct.unpack("<Q", f.read(8))  # tensor count (unused)
        (kv_count,) = struct.unpack("<Q", f.read(8))

        arch = None
        pending_ctx = None  # ctx value seen before we knew the architecture
        for _ in range(kv_count):
            key = _read_str(f).decode("utf-8", "replace")
            (value_type,) = struct.unpack("<I", f.read(4))
            if value_type == _ARRAY:
                (elem_type,) = struct.unpack("<I", f.read(4))
                (length,) = struct.unpack("<Q", f.read(8))
                if elem_type == _STRING:
                    for _ in range(length):
                        f.seek(struct.unpack("<Q", f.read(8))[0], 1)
                else:
                    f.seek(_SIZE[elem_type] * length, 1)
                continue

            if key == "general.architecture":
                arch = _read_scalar(f, value_type)
            elif key.endswith(".context_length"):
                ctx = _read_scalar(f, value_type)
                if arch and key == f"{arch}.context_length":
                    return int(ctx)
                pending_ctx = int(ctx)
            else:
                _skip_scalar(f, value_type)

            if arch and pending_ctx is not None:
                return pending_ctx
    return pending_ctx
