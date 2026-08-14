"""Canonical JSON encoding — SPEC.md §7.

``json.dumps`` satisfies the rules given the right flags, so this is a validator
plus a thin wrapper. Kotlin hand-writes the equivalent encoder.
"""

from __future__ import annotations

import json
from typing import Any

from .errors import MetaValueError

_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


def validate_meta(meta: dict[str, Any], _path: str = "meta") -> None:
    """Reject anything the other implementation could not reproduce byte-for-byte.

    Checked at write time; by read time the bad file already exists.
    """
    for key, value in meta.items():
        if not isinstance(key, str):
            raise MetaValueError(_path, key)
        _validate_value(value, f"{_path}.{key}")


def _validate_value(value: Any, path: str) -> None:
    if value is None or isinstance(value, str):
        return
    # bool before int: bool is a subclass of int in Python.
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        if not _INT64_MIN <= value <= _INT64_MAX:
            raise MetaValueError(path, value)
        return
    if isinstance(value, list):
        for i, item in enumerate(value):
            _validate_value(item, f"{path}[{i}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise MetaValueError(path, key)
            _validate_value(item, f"{path}.{key}")
        return
    raise MetaValueError(path, value)


def encode(obj: dict[str, Any], *, sort_keys: bool = False) -> bytes:
    """Encode one object to canonical UTF-8 JSON with no trailing newline.

    ``sort_keys`` is False where SPEC.md fixes the order and True for
    user-supplied ``meta``. Python sorts ``str`` by code point, which is what §7
    requires; the JVM's natural ordering is by UTF-16 code unit.
    """
    return json.dumps(
        obj,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=sort_keys,
        allow_nan=False,
    ).encode("utf-8")


def encode_string(value: str) -> bytes:
    """A single JSON string literal, escaped per §7 rule 3."""
    return json.dumps(value, ensure_ascii=False).encode("utf-8")


def encode_document(doc_id: str, text: str, meta: dict[str, Any], content_hash: str) -> bytes:
    """One line of docs.jsonl, without the terminating LF. SPEC.md §4.

    Assembled piece by piece so the outer key order is fixed here while
    ``meta``'s keys sort independently.
    """
    validate_meta(meta)
    return b"".join(
        [
            b'{"id":',
            encode_string(doc_id),
            b',"text":',
            encode_string(text),
            b',"meta":',
            encode(meta, sort_keys=True),
            b',"hash":',
            encode_string(content_hash),
            b"}",
        ]
    )


def decode_line(line: bytes) -> dict[str, Any]:
    parsed: dict[str, Any] = json.loads(line.decode("utf-8"))
    return parsed
