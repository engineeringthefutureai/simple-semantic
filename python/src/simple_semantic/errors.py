"""Errors. Each carries the specific ids, paths or sizes involved."""

from __future__ import annotations


class SimpleSemanticError(Exception):
    """Base for everything this library raises deliberately."""


class EmbedderMismatchError(SimpleSemanticError):
    """The index was written by a different embedder than the one configured.

    SPEC.md §2.1.
    """

    def __init__(self, path: str, index_id: str, configured_id: str) -> None:
        self.index_id = index_id
        self.configured_id = configured_id
        super().__init__(
            f"index at {path!s} was written with embedder_id "
            f"{index_id!r} but the configured embedder is {configured_id!r}. "
            f"Scores across two embedding spaces are meaningless. "
            f"Re-index with the new embedder, or open with the original one."
        )


class FormatVersionError(SimpleSemanticError):
    def __init__(self, path: str, found: object, supported: int) -> None:
        super().__init__(
            f"index at {path!s} declares format_version {found!r}; "
            f"this build supports version {supported}"
        )


class CorruptIndexError(SimpleSemanticError):
    """A structural check from SPEC.md §2.2 failed."""


class MetaValueError(SimpleSemanticError):
    """A meta value is outside the types the format can round-trip. SPEC.md §7.2."""

    def __init__(self, key_path: str, value: object) -> None:
        super().__init__(
            f"meta value at {key_path!r} has unsupported type "
            f"{type(value).__name__} ({value!r}). Permitted: str, bool, None, "
            f"int (64-bit), list, dict with str keys. Floats are not permitted "
            f"in format v1 — store the value as a string."
        )


class ExtractionError(SimpleSemanticError):
    """A class could not be turned into indexable documents."""
