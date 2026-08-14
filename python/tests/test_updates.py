"""Update semantics. SPEC.md §6.

Several of these are written specifically to catch the class of defect found
in the sibling project `simple-fts`, where an update left stale copies of a
document alive. The append-only row layout makes that shape of bug
structurally impossible here — these tests prove the claim rather than
assuming it.
"""

from __future__ import annotations

import numpy as np
import pytest

from simple_semantic import Document, HashingEmbedder, SemanticIndex
from simple_semantic import format as fmt


async def test_upsert_returns_only_the_new_version(index: SemanticIndex) -> None:
    """The `simple-fts` bug: a term shared by both versions returned the stale copy."""
    shared = "shared vocabulary appears in both revisions"
    await index.add_all([Document(id="doc", text=f"{shared} original wording")])
    await index.upsert(Document(id="doc", text=f"{shared} revised wording"))

    results = await index.search(shared, k=10)
    assert len(results) == 1
    assert "revised" in results[0].text
    assert "original" not in results[0].text


async def test_result_ids_are_distinct(index: SemanticIndex) -> None:
    """A query matching many rows must never return the same id twice."""
    for revision in range(5):
        await index.upsert(Document(id="doc", text=f"common terms revision {revision}"))
    await index.add_all([Document(id=f"other{i}", text="common terms") for i in range(3)])

    results = await index.search("common terms", k=20)
    ids = [r.id for r in results]
    assert len(ids) == len(set(ids))


async def test_live_count_after_n_upserts_of_one_id_is_one(index: SemanticIndex) -> None:
    for revision in range(10):
        await index.upsert(Document(id="doc", text=f"revision {revision}"))
    assert index.live_count() == 1
    assert index.size() == 10  # append-only: every revision is still a row


async def test_tombstoned_row_never_appears_even_as_nearest(index: SemanticIndex) -> None:
    """The deleted row is an exact match for the query; it must still not surface."""
    query = "precisely this exact phrase"
    await index.add_all(
        [
            Document(id="target", text=query),
            Document(id="other", text="a completely unrelated sentence about ferrets"),
        ]
    )
    before = await index.search(query, k=5)
    assert before[0].id == "target"

    assert index.delete("target") is True
    after = await index.search(query, k=5)
    assert [r.id for r in after] == ["other"]


async def test_delete_of_unknown_id_is_a_noop(index: SemanticIndex) -> None:
    await index.add_all([Document(id="a", text="hello")])
    assert index.delete("never-added") is False
    assert index.live_count() == 1


async def test_readding_identical_document_is_a_noop(index: SemanticIndex) -> None:
    """Driven by the content hash. This is the API cost control. SPEC.md §4.1."""
    doc = Document(id="a", text="unchanged text", meta={"k": "v"})
    first = await index.add_all([doc])
    second = await index.add_all([doc])

    assert first.added == 1 and first.skipped == 0
    assert second.added == 0 and second.replaced == 0 and second.skipped == 1
    assert index.size() == 1  # no second row appended


async def test_duplicate_id_within_one_batch_resolves_to_the_last(index: SemanticIndex) -> None:
    await index.add_all(
        [
            Document(id="a", text="first version"),
            Document(id="a", text="second version"),
        ]
    )
    assert index.live_count() == 1
    doc = index.get("a")
    assert doc is not None and doc.text == "second version"


async def test_append_only_never_mutates_an_existing_row(index: SemanticIndex) -> None:
    await index.add_all([Document(id="a", text="original")])
    original_row = np.array(
        np.memmap(index.path / fmt.VECTORS, dtype="<f4", mode="r", shape=(1, 64))[0]
    )

    await index.upsert(Document(id="a", text="replacement"))
    after = np.memmap(index.path / fmt.VECTORS, dtype="<f4", mode="r", shape=(2, 64))
    assert np.array_equal(np.asarray(after[0]), original_row)


async def test_compaction_drops_tombstones_and_preserves_search(index: SemanticIndex) -> None:
    await index.add_all([Document(id=f"d{i}", text=f"document about topic {i}") for i in range(6)])
    index.delete("d1")
    index.delete("d4")
    await index.upsert(Document(id="d0", text="document about topic zero, revised"))

    before = [(r.id, round(r.score, 9)) for r in await index.search("document about topic", k=10)]
    assert index.size() == 7 and index.live_count() == 4

    dropped = index.compact()
    assert dropped == 3
    assert index.size() == 4 and index.live_count() == 4
    assert not (index.path / fmt.TOMBSTONES).exists()

    after = [(r.id, round(r.score, 9)) for r in await index.search("document about topic", k=10)]
    assert before == after


async def test_compaction_is_a_noop_when_nothing_is_deleted(index: SemanticIndex) -> None:
    await index.add_all([Document(id="a", text="hello")])
    assert index.compact() == 0
    assert index.size() == 1


async def test_state_survives_close_and_reopen(index: SemanticIndex, tmp_path) -> None:
    await index.add_all([Document(id=f"d{i}", text=f"text {i}", meta={"n": i}) for i in range(4)])
    index.delete("d2")
    await index.upsert(Document(id="d0", text="text zero revised"))
    expected = [(r.id, r.score) for r in await index.search("text", k=10)]
    path = index.path
    index.close()

    with SemanticIndex.open(path, HashingEmbedder(dimension=64, seed=0)) as reopened:
        assert reopened.live_count() == 3
        assert reopened.contains("d0") and not reopened.contains("d2")
        got = reopened.get("d0")
        assert got is not None and got.text == "text zero revised"
        assert [(r.id, r.score) for r in await reopened.search("text", k=10)] == expected


async def test_compaction_survives_a_crash_after_staging(index: SemanticIndex) -> None:
    """The staging directory is complete before anything in place is touched."""
    await index.add_all([Document(id=f"d{i}", text=f"t{i}") for i in range(3)])
    index.delete("d1")

    staging = index.path.parent / f"{index.path.name}.compact.tmp"
    staging.mkdir()
    (staging / "leftover").write_text("debris from an interrupted run")

    # A leftover staging directory must not corrupt or block the next attempt.
    assert index.compact() == 1
    assert index.live_count() == 2
    assert not staging.exists()


@pytest.mark.parametrize("k", [0, 1, 3, 50])
async def test_k_larger_than_live_count_returns_all_live_rows(index: SemanticIndex, k: int) -> None:
    await index.add_all([Document(id=f"d{i}", text=f"document {i}") for i in range(3)])
    results = await index.search("document", k=k)
    assert len(results) == min(k, 3)
