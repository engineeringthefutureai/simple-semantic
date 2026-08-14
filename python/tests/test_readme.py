"""The README examples, executed.

`simple-fts` shipped an extractor that crashed on exactly the nullable field
its own README declared, because the example was prose and the tests were code.
Here the example *is* a test: if the API moves, this fails before anyone reads
a stale snippet.

Keep these in sync with README.md by hand — the only difference permitted is
the index path, which points at a tmp directory instead of ./notes.index.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

from simple_semantic import (
    Document,
    HashingEmbedder,
    SemanticId,
    SemanticIndex,
    SemanticIndexed,
    SemanticMeta,
    from_dataclass,
    from_lambdas,
)


async def test_readme_quickstart(tmp_path: Path) -> None:
    embedder = HashingEmbedder(dimension=256)
    index = SemanticIndex.create(tmp_path / "notes.index", embedder)

    await index.add_all(
        [
            Document(
                id="n1",
                text="Cosine similarity over a dense matrix.",
                meta={"source": "notes", "tags": ["ir"]},
            ),
            Document(
                id="n2",
                text="Sourdough needs a mature starter.",
                meta={"source": "recipes"},
            ),
        ]
    )

    hits = await index.search("vector similarity", k=5)
    assert [f"{hit.score:+.4f}  {hit.id}" for hit in hits]

    await index.upsert(Document(id="n1", text="Revised text."))
    index.delete("n2")
    index.compact()

    assert index.live_count() == 1
    assert index.size() == 1  # both tombstoned rows are gone after compaction
    revised = index.get("n1")
    assert revised is not None and revised.text == "Revised text."
    index.close()


@dataclass
class Note:
    id: Annotated[str, SemanticId]
    title: Annotated[str, SemanticIndexed]
    body: Annotated[str | None, SemanticIndexed]
    tags: Annotated[list[str], SemanticMeta] = field(default_factory=list)


def test_readme_extraction() -> None:
    notes = [Note(id="n1", title="Embeddings", body=None, tags=["ml"])]
    docs = from_dataclass(notes)
    assert docs[0].id == "n1"
    # The nullable body is skipped rather than raising.
    assert docs[0].text == "Embeddings"

    rows = [{"key": "r1", "body": "some text"}]
    docs = from_lambdas(rows, id_of=lambda r: r["key"], text_of=lambda r: r["body"])
    assert docs[0].id == "r1"
