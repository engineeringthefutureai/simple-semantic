"""The on-disk format. SPEC.md §2-§7."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from simple_semantic import (
    CorruptIndexError,
    Document,
    EmbedderMismatchError,
    HashingEmbedder,
    MetaValueError,
    SemanticIndex,
    content_hash,
    normalize_row,
)
from simple_semantic import format as fmt
from simple_semantic.canonical_json import encode, encode_document


async def test_files_have_the_shapes_the_manifest_declares(index: SemanticIndex) -> None:
    await index.add_all([Document(id=f"d{i}", text=f"document number {i}") for i in range(5)])
    path = index.path
    manifest = fmt.read_manifest(path)

    assert manifest.row_count == 5
    assert manifest.live_count == 5
    assert (path / fmt.VECTORS).stat().st_size == 5 * manifest.dimension * 4
    assert (path / fmt.OFFSETS).stat().st_size == 6 * 8
    # No deletions, so the tombstone file need not exist at all. SPEC.md §6.
    assert not (path / fmt.TOMBSTONES).exists()


async def test_docs_jsonl_line_n_matches_vector_row_n(index: SemanticIndex) -> None:
    """The core invariant of SPEC.md §1."""
    texts = ["alpha beta", "gamma delta", "epsilon zeta"]
    await index.add_all([Document(id=f"d{i}", text=t) for i, t in enumerate(texts)])

    lines = (index.path / fmt.DOCS).read_bytes().splitlines()
    vectors = np.memmap(
        index.path / fmt.VECTORS, dtype="<f4", mode="r", shape=(3, index.manifest.dimension)
    )
    for row, line in enumerate(lines):
        obj = json.loads(line)
        assert obj["text"] == texts[row]
        expected = index.embedder.embed_text(texts[row])
        assert np.array_equal(np.asarray(vectors[row]), expected)


async def test_every_stored_row_is_unit_norm(index: SemanticIndex) -> None:
    await index.add_all(
        [
            Document(id="a", text="the quick brown fox"),
            Document(id="b", text="jumps over the lazy dog"),
            Document(id="c", text="x"),
        ]
    )
    vectors = np.memmap(
        index.path / fmt.VECTORS, dtype="<f4", mode="r", shape=(3, index.manifest.dimension)
    )
    for row in range(3):
        norm = float(np.linalg.norm(np.asarray(vectors[row], dtype=np.float64)))
        assert abs(norm - 1.0) < 1e-5, f"row {row} has norm {norm}"


async def test_offsets_index_exactly_the_lines(index: SemanticIndex) -> None:
    await index.add_all([Document(id=f"d{i}", text=f"text {i}") for i in range(4)])
    raw = (index.path / fmt.DOCS).read_bytes()
    offsets = fmt.read_offsets(index.path, 4)

    assert int(offsets[-1]) == len(raw)
    for row in range(4):
        line = raw[int(offsets[row]) : int(offsets[row + 1])]
        assert line.endswith(b"\n")
        assert json.loads(line)["id"] == f"d{row}"


def test_vectors_file_is_little_endian(tmp_path: Path) -> None:
    """SPEC.md §3. The JVM defaults to big-endian, so this must be pinned.

    Written as a byte-level assertion rather than a round-trip, because a
    round-trip through one implementation passes happily with either
    convention — it is only the *other* implementation that notices.
    """
    row = np.array([1.0, -2.0, 0.5], dtype="<f4")
    path = tmp_path / "v.f32"
    path.write_bytes(row.tobytes())
    raw = path.read_bytes()
    # 1.0f is 0x3F800000; little-endian on disk is 00 00 80 3F.
    assert raw[0:4] == bytes([0x00, 0x00, 0x80, 0x3F])
    assert np.frombuffer(raw, dtype="<f4")[1] == -2.0


def test_tombstone_bits_are_lsb_first() -> None:
    """SPEC.md §6. MSB-first parses without error and means something else."""
    bits = fmt.Tombstones(10)
    bits.mark_deleted(0)
    bits.mark_deleted(9)
    raw = bits.to_bytes()
    assert len(raw) == 2
    assert raw[0] == 0b0000_0001  # row 0 -> least significant bit
    assert raw[1] == 0b0000_0010  # row 9 -> bit 1 of byte 1
    live = bits.live_mask().tolist()
    assert live == [False] + [True] * 8 + [False]


def test_content_hash_is_stable_and_separated() -> None:
    """SPEC.md §4.1. The NUL separators make the concatenation unambiguous."""
    assert content_hash("abc", "e", "c") != content_hash("ab", "ce", "c")
    assert content_hash("abc", "e", "c") == content_hash("abc", "e", "c")
    # Changing the embedder invalidates the vector even when the text is equal.
    assert content_hash("abc", "e1", "c") != content_hash("abc", "e2", "c")


def test_canonical_json_has_no_whitespace_and_sorted_meta() -> None:
    assert encode({"b": 1, "a": 2}, sort_keys=True) == b'{"a":2,"b":1}'
    line = encode_document("id", "text", {"z": 1, "a": "x"}, "hash")
    assert line == b'{"id":"id","text":"text","meta":{"a":"x","z":1},"hash":"hash"}'


def test_canonical_json_emits_non_ascii_literally() -> None:
    """Escaping non-ASCII would still be valid JSON and would still break byte-identity."""
    line = encode_document("id", "Привет 日本語 café", {}, "h")
    assert "Привет 日本語 café".encode() in line
    assert b"\\u" not in line


def test_meta_rejects_floats(index: SemanticIndex) -> None:
    """SPEC.md §7.2 — not because floats are hard, because their decimal form is not portable."""
    with pytest.raises(MetaValueError) as caught:
        encode_document("id", "text", {"score": 0.5}, "h")
    assert "score" in str(caught.value)


async def test_meta_is_validated_before_the_embedder_is_called(index: SemanticIndex) -> None:
    """Validating after embedding would mean paying for a call that then fails."""
    calls = 0
    original = index.embedder.embed_documents

    async def counting(texts: list[str]) -> np.ndarray:
        nonlocal calls
        calls += 1
        return await original(texts)

    index.embedder.embed_documents = counting  # type: ignore[method-assign]
    with pytest.raises(MetaValueError):
        await index.add_all([Document(id="a", text="hello", meta={"score": 0.5})])
    assert calls == 0


async def test_open_refuses_a_different_embedder(index: SemanticIndex) -> None:
    """SPEC.md §2.1: the single most important correctness rule in the project."""
    await index.add_all([Document(id="a", text="hello")])
    index.close()

    other = HashingEmbedder(dimension=64, seed=1)
    with pytest.raises(EmbedderMismatchError) as caught:
        SemanticIndex.open(index.path, other)

    message = str(caught.value)
    assert "hashing-0@64" in message
    assert "hashing-1@64" in message


async def test_open_refuses_a_truncated_vectors_file(index: SemanticIndex) -> None:
    await index.add_all([Document(id=f"d{i}", text=f"t{i}") for i in range(3)])
    index.close()

    path = index.path / fmt.VECTORS
    raw = path.read_bytes()
    path.write_bytes(raw[:-8])

    with pytest.raises(CorruptIndexError) as caught:
        SemanticIndex.open(index.path, HashingEmbedder(dimension=64, seed=0))
    assert "truncated" in str(caught.value)


async def test_open_refuses_an_unknown_format_version(index: SemanticIndex) -> None:
    index.close()
    manifest = json.loads((index.path / fmt.MANIFEST).read_text())
    manifest["format_version"] = 99
    (index.path / fmt.MANIFEST).write_text(json.dumps(manifest))

    with pytest.raises(Exception) as caught:
        SemanticIndex.open(index.path, HashingEmbedder(dimension=64, seed=0))
    assert "99" in str(caught.value)


def test_the_version_is_checked_before_the_rest_of_the_manifest(tmp_path: Path) -> None:
    """A v2 manifest is under no obligation to carry v1's fields.

    Decoding the whole object first would report a missing field instead of the
    version number that explains why the field is missing.
    """
    (tmp_path / fmt.MANIFEST).write_text(
        json.dumps({"format_version": 2, "segments": [{"embedder": "hashing-0@64"}]})
    )
    with pytest.raises(Exception) as caught:
        fmt.read_manifest(tmp_path)
    assert "2" in str(caught.value)


async def test_offsets_must_increase(index: SemanticIndex) -> None:
    """SPEC.md §5. A non-increasing pair would be read as a negative length."""
    await index.add_all([Document(id=f"d{i}", text=f"document number {i}") for i in range(4)])
    path = index.path
    index.close()

    offsets = bytearray((path / fmt.OFFSETS).read_bytes())
    offsets[8:16] = (0).to_bytes(8, "little")
    (path / fmt.OFFSETS).write_bytes(bytes(offsets))

    with pytest.raises(CorruptIndexError) as caught:
        SemanticIndex.open(path, HashingEmbedder(dimension=64, seed=0))
    assert "not greater than" in str(caught.value)


def test_normalization_handles_the_zero_vector() -> None:
    """SPEC.md §3.1: 0/0 would put a NaN in the matrix that poisons every dot product."""
    out = normalize_row(np.zeros(8, dtype=np.float32))
    assert not np.isnan(out).any()
    assert out[0] == 1.0
    assert abs(float(np.linalg.norm(out)) - 1.0) < 1e-6


def test_normalization_is_idempotent() -> None:
    raw = np.array([3.0, 4.0, 0.0], dtype=np.float32)
    once = normalize_row(raw)
    twice = normalize_row(once)
    assert np.array_equal(once, twice)


def test_manifest_decoding_names_a_missing_field(tmp_path: Path) -> None:
    """Declarative decoding, so the error names the field rather than raising KeyError."""
    manifest = {
        "format_version": 1,
        "embedder_id": "hashing-0@64",
        "dimension": 64,
        "chunker_id": "none",
        "row_count": 0,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    }  # live_count omitted
    path = tmp_path / fmt.MANIFEST
    path.write_text(json.dumps(manifest))
    with pytest.raises(CorruptIndexError) as caught:
        fmt.read_manifest(tmp_path)
    assert "live_count" in str(caught.value)


def test_manifest_decoding_names_a_mistyped_field(tmp_path: Path) -> None:
    manifest = {
        "format_version": 1,
        "embedder_id": "hashing-0@64",
        "dimension": "sixty-four",
        "chunker_id": "none",
        "row_count": 0,
        "live_count": 0,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    }
    (tmp_path / fmt.MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(CorruptIndexError) as caught:
        fmt.read_manifest(tmp_path)
    assert "dimension" in str(caught.value)
    assert "str" in str(caught.value)


async def test_a_document_line_missing_text_is_reported(index: SemanticIndex) -> None:
    """Every line is decoded at open, so a corrupt one fails there."""
    await index.add_all([Document(id="a", text="hello")])
    index.close()
    (index.path / fmt.DOCS).write_bytes(b'{"id":"a","meta":{},"hash":"x"}\n')

    with pytest.raises(CorruptIndexError) as caught:
        SemanticIndex.open(index.path, HashingEmbedder(dimension=64))
    assert "text" in str(caught.value)
