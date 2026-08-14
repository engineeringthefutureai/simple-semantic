"""The shapes this package reads, declared once.

Encoding stays hand-assembled — SPEC.md §7 pins a canonical form — but decoding
has no such constraint, so the wire shapes are dataclasses and :func:`decode`
maps a parsed JSON object onto them. Field names, types and defaults live in one
declaration instead of being spelled out again at each ``obj["..."]``.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields
from typing import Any, TypeVar

from .errors import SimpleSemanticError

T = TypeVar("T")

#: ``from __future__ import annotations`` makes every annotation a string, so
#: the check is on the leading token. Anything unlisted is accepted as-is.
_RUNTIME_TYPES: dict[str, type | tuple[type, ...]] = {
    "str": str,
    "int": int,
    "bool": bool,
    "float": (int, float),
    "dict": dict,
    "list": list,
}


def decode(cls: type[T], obj: dict[str, Any], source: str) -> T:
    """Map a parsed JSON object onto a wire dataclass.

    Raises a :class:`SimpleSemanticError` naming the field and what was wrong,
    rather than letting a ``KeyError`` or a silent coercion out.
    """
    values: dict[str, Any] = {}
    for spec in fields(cls):  # type: ignore[arg-type]
        if spec.name not in obj:
            if spec.default is not MISSING:
                values[spec.name] = spec.default
            elif spec.default_factory is not MISSING:
                values[spec.name] = spec.default_factory()
            else:
                raise SimpleSemanticError(f"{source}: missing {spec.name!r}")
            continue

        value = obj[spec.name]
        expected = _RUNTIME_TYPES.get(str(spec.type).split("[")[0].strip())
        if expected is not None and not _matches(value, expected):
            raise SimpleSemanticError(
                f"{source}: {spec.name!r} is {type(value).__name__}, expected {spec.type}"
            )
        values[spec.name] = value
    return cls(**values)


def _matches(value: Any, expected: type | tuple[type, ...]) -> bool:
    # bool is a subclass of int, so an unguarded isinstance would accept True
    # where a count is required.
    if expected is not bool and isinstance(value, bool):
        return False
    return isinstance(value, expected)


@dataclass(frozen=True)
class ManifestWire:
    """`manifest.json`. SPEC.md §2."""

    format_version: int
    embedder_id: str
    dimension: int
    chunker_id: str
    row_count: int
    live_count: int
    created_at: str
    updated_at: str
    normalized: bool = True
    hash_algorithm: str = "sha256"


@dataclass(frozen=True)
class DocumentWire:
    """One line of `docs.jsonl`. SPEC.md §4."""

    id: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)
    hash: str = ""


@dataclass(frozen=True)
class FixtureRecordWire:
    key: str
    vector: list[float]
    id: str = ""
    file: str = ""
    prompt: str = ""
    target: str = ""


@dataclass(frozen=True)
class FixtureWire:
    """The replay fixture. SPEC.md appendix B."""

    embedder_id: str
    dimension: int
    documents: list[FixtureRecordWire]
    queries: list[FixtureRecordWire]
    normalized: bool = False

    @staticmethod
    def from_json(obj: dict[str, Any], source: str) -> FixtureWire:
        base = decode(FixtureWire, {**obj, "documents": [], "queries": []}, source)
        return FixtureWire(
            embedder_id=base.embedder_id,
            dimension=base.dimension,
            normalized=base.normalized,
            documents=[
                decode(FixtureRecordWire, record, f"{source}: documents")
                for record in obj.get("documents", [])
            ],
            queries=[
                decode(FixtureRecordWire, record, f"{source}: queries")
                for record in obj.get("queries", [])
            ],
        )
