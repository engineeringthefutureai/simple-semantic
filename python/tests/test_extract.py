"""Document extraction. Both entry points are public — see extract.py."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated

import pytest

from simple_semantic import (
    ExtractionError,
    SemanticId,
    SemanticIndexed,
    SemanticMeta,
    from_dataclass,
    from_lambdas,
)


@dataclass
class Note:
    id: Annotated[str, SemanticId]
    title: Annotated[str, SemanticIndexed]
    body: Annotated[str | None, SemanticIndexed]
    tags: Annotated[list[str], SemanticMeta] = field(default_factory=list)


def test_annotated_extraction() -> None:
    docs = from_dataclass([Note(id="n1", title="Embeddings", body="Dense vectors.", tags=["ml"])])
    assert len(docs) == 1
    assert docs[0].id == "n1"
    assert docs[0].text == "Embeddings\n\nDense vectors."
    assert docs[0].meta == {"tags": ["ml"]}


def test_nullable_indexed_field_does_not_throw() -> None:
    """`simple-fts` cast with `value as String` and crashed on exactly the
    nullable field its own README example declared."""
    docs = from_dataclass([Note(id="n1", title="Title only", body=None)])
    assert docs[0].text == "Title only"


def test_missing_id_annotation_names_the_annotation_and_the_class() -> None:
    @dataclass
    class NoId:
        title: Annotated[str, SemanticIndexed]

    with pytest.raises(ExtractionError) as caught:
        from_dataclass([NoId(title="x")])
    message = str(caught.value)
    assert "NoId" in message
    assert "SemanticId" in message


def test_missing_indexed_annotation_is_an_error() -> None:
    @dataclass
    class NoText:
        id: Annotated[str, SemanticId]

    with pytest.raises(ExtractionError) as caught:
        from_dataclass([NoText(id="x")])
    assert "SemanticIndexed" in str(caught.value)


def test_two_id_fields_is_an_error() -> None:
    @dataclass
    class TwoIds:
        a: Annotated[str, SemanticId]
        b: Annotated[str, SemanticId]
        t: Annotated[str, SemanticIndexed]

    with pytest.raises(ExtractionError) as caught:
        from_dataclass([TwoIds(a="1", b="2", t="t")])
    assert "exactly one" in str(caught.value)


def test_null_id_is_an_error_naming_the_field() -> None:
    @dataclass
    class NullableId:
        id: Annotated[str | None, SemanticId]
        t: Annotated[str, SemanticIndexed]

    with pytest.raises(ExtractionError) as caught:
        from_dataclass([NullableId(id=None, t="t")])
    assert "NullableId.id" in str(caught.value)


def test_non_dataclass_points_at_the_lambda_factory() -> None:
    class Plain:
        pass

    with pytest.raises(ExtractionError) as caught:
        from_dataclass([Plain()])
    assert "from_lambdas" in str(caught.value)


def test_lambda_factory_is_public_and_needs_no_annotations() -> None:
    rows = [{"key": "r1", "body": "some text", "source": "db"}]
    docs = from_lambdas(
        rows,
        id_of=lambda r: r["key"],
        text_of=lambda r: r["body"],
        meta_of=lambda r: {"source": r["source"]},
    )
    assert docs[0].id == "r1"
    assert docs[0].text == "some text"
    assert docs[0].meta == {"source": "db"}


def test_lambda_factory_meta_is_optional() -> None:
    docs = from_lambdas([("a", "text")], id_of=lambda t: t[0], text_of=lambda t: t[1])
    assert docs[0].meta == {}


def test_empty_input_yields_no_documents() -> None:
    assert from_dataclass([]) == []
    assert from_lambdas([], id_of=str, text_of=str) == []


async def test_extracted_documents_index_and_search(index) -> None:
    notes = [
        Note(id="n1", title="Vector search", body="Cosine similarity over a matrix.", tags=["ir"]),
        Note(id="n2", title="Bread", body=None, tags=["food"]),
    ]
    await index.add_all(from_dataclass(notes))
    results = await index.search("Cosine similarity over a matrix.", k=1)
    assert results[0].id == "n1"
    assert results[0].meta == {"tags": ["ir"]}
