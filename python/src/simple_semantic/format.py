"""The four files. SPEC.md §2-§6.

This module knows the byte layout and nothing about search. Everything here
has a direct clause in SPEC.md; where the code makes a choice the spec does
not force, the comment says which.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from . import canonical_json
from .errors import CorruptIndexError, FormatVersionError

FORMAT_VERSION = 1

MANIFEST = "manifest.json"
VECTORS = "vectors.f32"
DOCS = "docs.jsonl"
OFFSETS = "offsets.bin"
TOMBSTONES = "tombstones.bits"

#: SPEC.md §3 pins little-endian binary32, independent of host byte order.
VECTOR_DTYPE = np.dtype("<f4")
#: SPEC.md §5 pins unsigned 64-bit little-endian.
OFFSET_DTYPE = np.dtype("<u8")


def utc_now() -> str:
    """RFC 3339, UTC, second precision, Z suffix. SPEC.md §2."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def content_hash(text: str, embedder_id: str, chunker_id: str) -> str:
    """SPEC.md §4.1: sha256(text || 0x00 || embedder_id || 0x00 || chunker_id).

    The NUL separators are unambiguous because none of the three inputs may
    contain a NUL byte. Without them, ("ab", "c") and ("a", "bc") would hash
    alike, and an id containing the delimiter would let one document
    impersonate another's hash.
    """
    digest = hashlib.sha256()
    digest.update(text.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(embedder_id.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(chunker_id.encode("utf-8"))
    return digest.hexdigest()


@dataclass
class Manifest:
    """The header for the other three files. SPEC.md §2."""

    embedder_id: str
    dimension: int
    chunker_id: str
    row_count: int
    live_count: int
    created_at: str
    updated_at: str
    format_version: int = FORMAT_VERSION
    normalized: bool = True
    hash_algorithm: str = "sha256"

    def to_bytes(self) -> bytes:
        # Key order is fixed by SPEC.md §2 and this literal is where it is
        # fixed. Do not sort.
        return canonical_json.encode(
            {
                "format_version": self.format_version,
                "embedder_id": self.embedder_id,
                "dimension": self.dimension,
                "normalized": self.normalized,
                "chunker_id": self.chunker_id,
                "hash_algorithm": self.hash_algorithm,
                "row_count": self.row_count,
                "live_count": self.live_count,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
            }
        )

    @staticmethod
    def from_bytes(raw: bytes, path: str) -> Manifest:
        try:
            obj: dict[str, Any] = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise CorruptIndexError(f"{path}: manifest is not valid UTF-8 JSON ({exc})") from exc

        version = obj.get("format_version")
        if version != FORMAT_VERSION:
            raise FormatVersionError(path, version, FORMAT_VERSION)

        missing = [
            key
            for key in (
                "embedder_id",
                "dimension",
                "chunker_id",
                "row_count",
                "live_count",
                "created_at",
                "updated_at",
            )
            if key not in obj
        ]
        if missing:
            raise CorruptIndexError(f"{path}: manifest is missing keys {missing}")

        return Manifest(
            embedder_id=str(obj["embedder_id"]),
            dimension=int(obj["dimension"]),
            chunker_id=str(obj["chunker_id"]),
            row_count=int(obj["row_count"]),
            live_count=int(obj["live_count"]),
            created_at=str(obj["created_at"]),
            updated_at=str(obj["updated_at"]),
            format_version=int(version),
            normalized=bool(obj.get("normalized", True)),
            hash_algorithm=str(obj.get("hash_algorithm", "sha256")),
        )


class Tombstones:
    """A bit per row, LSB-first within each byte. SPEC.md §6.

    LSB-first is stated in the spec and repeated here because MSB-first is an
    equally common convention that produces a file of the same size which
    parses without error and disagrees about which rows are deleted.
    """

    def __init__(self, row_count: int, raw: bytes | None = None) -> None:
        needed = (row_count + 7) // 8
        if raw is None:
            self._bits = bytearray(needed)
        else:
            if len(raw) != needed:
                raise CorruptIndexError(
                    f"{TOMBSTONES} is {len(raw)} bytes but row_count {row_count} "
                    f"requires exactly {needed}"
                )
            self._bits = bytearray(raw)
        self._row_count = row_count
        self._deleted = sum(bin(byte).count("1") for byte in self._bits)

    def __len__(self) -> int:
        return self._row_count

    @property
    def deleted_count(self) -> int:
        return self._deleted

    def is_deleted(self, row: int) -> bool:
        if not 0 <= row < self._row_count:
            raise IndexError(f"row {row} out of range for {self._row_count} rows")
        return bool(self._bits[row >> 3] & (1 << (row & 7)))

    def mark_deleted(self, row: int) -> bool:
        """Set the bit. Returns True if this call changed it."""
        if self.is_deleted(row):
            return False
        self._bits[row >> 3] |= 1 << (row & 7)
        self._deleted += 1
        return True

    def grow_to(self, row_count: int) -> None:
        if row_count < self._row_count:
            raise ValueError(f"cannot shrink tombstones from {self._row_count} to {row_count}")
        self._row_count = row_count
        needed = (row_count + 7) // 8
        if len(self._bits) < needed:
            self._bits.extend(bytes(needed - len(self._bits)))

    def live_mask(self) -> np.ndarray:
        """Boolean array, True where the row is live.

        ``np.unpackbits`` with bitorder="little" is exactly the LSB-first
        convention of §6, so this is one call rather than a Python loop over
        every row.
        """
        if self._row_count == 0:
            return np.zeros(0, dtype=bool)
        bits = np.unpackbits(np.frombuffer(bytes(self._bits), dtype=np.uint8), bitorder="little")
        return ~bits[: self._row_count].astype(bool)

    def to_bytes(self) -> bytes:
        # Padding bits past row_count are already zero and stay that way:
        # mark_deleted range-checks, and grow_to appends zero bytes.
        return bytes(self._bits)

    def any_deleted(self) -> bool:
        return self._deleted > 0


def read_manifest(directory: Path) -> Manifest:
    path = directory / MANIFEST
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise CorruptIndexError(f"{path}: no {MANIFEST}; not a simple-semantic index") from exc
    return Manifest.from_bytes(raw, str(path))


def read_tombstones(directory: Path, row_count: int) -> Tombstones:
    path = directory / TOMBSTONES
    if not path.exists():
        # An absent file means no deletions. SPEC.md §6.
        return Tombstones(row_count)
    return Tombstones(row_count, path.read_bytes())


def read_offsets(directory: Path, row_count: int) -> np.ndarray:
    path = directory / OFFSETS
    raw = path.read_bytes()
    expected = (row_count + 1) * 8
    if len(raw) != expected:
        raise CorruptIndexError(
            f"{path} is {len(raw)} bytes but row_count {row_count} requires "
            f"exactly {expected} ((row_count + 1) * 8)"
        )
    return np.frombuffer(raw, dtype=OFFSET_DTYPE).copy()


def open_vectors(directory: Path, row_count: int, dimension: int) -> np.ndarray:
    """Memory-map vectors.f32 read-only. SPEC.md §3.

    The headerless layout is what makes this a single call with no parsing and
    no offset arithmetic.
    """
    path = directory / VECTORS
    size = path.stat().st_size
    expected = row_count * dimension * 4
    if size != expected:
        raise CorruptIndexError(
            f"{path} is {size} bytes but manifest declares row_count "
            f"{row_count} x dimension {dimension} x 4 = {expected}. "
            f"The index is truncated or was copied while being written."
        )
    if row_count == 0:
        return np.zeros((0, dimension), dtype=VECTOR_DTYPE)
    return np.memmap(path, dtype=VECTOR_DTYPE, mode="r", shape=(row_count, dimension))


def fsync_path(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)  # noqa: SIM115 - a raw fd, closed below
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_atomic(path: Path, data: bytes) -> None:
    """Write via a sibling temp file and rename, so a reader never sees a partial file."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)
    fsync_path(path.parent)
