"""Turning your own objects into :class:`Document` values.

Two entry points, both public:

- :func:`from_dataclass`, driven by field annotations, for the common case.
- :func:`from_lambdas`, taking plain callables, for everything else.

The lambda-based factory being public is deliberate. The sibling project
`simple-fts` had exactly this constructor but marked it ``internal``, with the
only public factory hard-wiring the annotation extractors — an extension point
that was built and then sealed off. Both doors are open here.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Sequence
from typing import Annotated, Any, TypeVar, get_args, get_origin, get_type_hints

from .errors import ExtractionError
from .index import Document

T = TypeVar("T")


class _SemanticId:
    """Marks the field holding the document id."""

    def __repr__(self) -> str:
        return "SemanticId"


class _SemanticIndexed:
    """Marks a field whose text is embedded."""

    def __repr__(self) -> str:
        return "SemanticIndexed"


class _SemanticMeta:
    """Marks a field stored as filterable metadata but not embedded."""

    def __repr__(self) -> str:
        return "SemanticMeta"


#: ``id: Annotated[str, SemanticId]``
SemanticId = _SemanticId()
#: ``title: Annotated[str | None, SemanticIndexed]``
SemanticIndexed = _SemanticIndexed()
#: ``tags: Annotated[list[str], SemanticMeta]``
SemanticMeta = _SemanticMeta()


def from_lambdas(
    items: Iterable[T],
    *,
    id_of: Callable[[T], str],
    text_of: Callable[[T], str],
    meta_of: Callable[[T], dict[str, Any]] | None = None,
) -> list[Document]:
    """Build documents from three plain callables.

    No annotations, no reflection, no requirement that the input be a
    dataclass — a dict, an ORM row, or a tuple all work.
    """
    documents: list[Document] = []
    for item in items:
        meta = meta_of(item) if meta_of is not None else {}
        documents.append(Document(id=id_of(item), text=text_of(item), meta=dict(meta)))
    return documents


def from_dataclass(items: Iterable[T], *, separator: str = "\n\n") -> list[Document]:
    """Build documents from ``Annotated`` dataclass fields.

    Nullable annotated fields are skipped rather than crashing. `simple-fts`
    cast with ``value as String`` and blew up on precisely the nullable field
    its own README example declared, which is a good reminder that the
    happy-path example and the test suite have to be the same code.
    """
    items = list(items)
    if not items:
        return []

    cls = type(items[0])
    id_field, text_fields, meta_fields = _resolve_fields(cls)

    documents: list[Document] = []
    for item in items:
        raw_id = getattr(item, id_field)
        if raw_id is None:
            raise ExtractionError(
                f"{cls.__name__}.{id_field} is annotated SemanticId but is None; "
                f"a document id is required"
            )
        parts = [
            str(getattr(item, name)) for name in text_fields if getattr(item, name) is not None
        ]
        meta = {name: getattr(item, name) for name in meta_fields}
        documents.append(
            Document(id=str(raw_id), text=separator.join(parts), meta=_clean_meta(meta))
        )
    return documents


def _clean_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """Drop None values so an absent field is absent rather than stored as null."""
    return {key: value for key, value in meta.items() if value is not None}


def _resolve_fields(cls: type) -> tuple[str, list[str], list[str]]:
    if not dataclasses.is_dataclass(cls):
        raise ExtractionError(
            f"{cls.__name__} is not a dataclass. Use from_lambdas() for arbitrary types."
        )

    hints = get_type_hints(cls, include_extras=True)
    id_field: str | None = None
    text_fields: list[str] = []
    meta_fields: list[str] = []

    for field in dataclasses.fields(cls):
        markers = _markers(hints.get(field.name))
        if any(isinstance(marker, _SemanticId) for marker in markers):
            if id_field is not None:
                raise ExtractionError(
                    f"{cls.__name__} annotates both {id_field!r} and {field.name!r} "
                    f"with SemanticId; exactly one field may be the id"
                )
            id_field = field.name
        if any(isinstance(marker, _SemanticIndexed) for marker in markers):
            text_fields.append(field.name)
        if any(isinstance(marker, _SemanticMeta) for marker in markers):
            meta_fields.append(field.name)

    if id_field is None:
        raise ExtractionError(
            f"{cls.__name__} has no field annotated with SemanticId. Annotate the "
            f"id field as Annotated[str, SemanticId], or use from_lambdas()."
        )
    if not text_fields:
        raise ExtractionError(
            f"{cls.__name__} has no field annotated with SemanticIndexed, so there "
            f"is nothing to embed. Annotate at least one text field."
        )
    return id_field, text_fields, meta_fields


def _markers(hint: Any) -> Sequence[Any]:
    if get_origin(hint) is Annotated:
        return get_args(hint)[1:]
    return ()
