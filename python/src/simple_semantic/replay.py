"""``ReplayEmbedder`` — serve recorded vectors instead of calling a model.

Reads vectors a real model produced once, recorded to a JSON fixture. That makes
retrieval-quality assertions runnable offline with identical numbers every run.
SPEC.md appendix B.

A miss is a loud error, never a zero vector: inventing a plausible answer would
turn "this text is not in the recording" into "retrieval quietly got worse".
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .errors import SimpleSemanticError
from .wire import FixtureRecordWire, FixtureWire


class ReplayMissError(SimpleSemanticError):
    """The text asked for is not in the recording.

    Names the mode too: the commonest cause is asking for a document embedding
    of something recorded only as a query, which is a different vector.
    """

    def __init__(self, mode: str, text: str, key: str, source: str) -> None:
        preview = text if len(text) <= 60 else text[:57] + "..."
        super().__init__(
            f"no recorded {mode} embedding for {preview!r} "
            f"(sha256 {key[:16]}...) in {source}. "
            f"Either the text changed since the fixture was generated, or it was "
            f"recorded as a {'query' if mode == 'document' else 'document'} instead."
        )


def content_key(text: str) -> str:
    """sha256 of the exact text that was embedded. Must match build_fixture.py."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ReplayEmbedder:
    """An :class:`~simple_semantic.embedder.Embedder` backed by a recording.

    ``produces_normalized`` comes from the fixture. The story fixture says
    ``False``, which is what exercises SPEC.md §3.1 against unnormalized input.
    """

    #: No network, so batching exists only to satisfy the interface.
    max_batch_size = 1024

    def __init__(
        self,
        embedder_id: str,
        dimension: int,
        documents: dict[str, np.ndarray],
        queries: dict[str, np.ndarray],
        *,
        produces_normalized: bool = False,
        source: str = "<memory>",
    ) -> None:
        self.id = embedder_id
        self.dimension = dimension
        self.produces_normalized = produces_normalized
        self._documents = documents
        self._queries = queries
        self._source = source

    @classmethod
    def from_file(cls, path: str | Path) -> ReplayEmbedder:
        """Load a fixture written by ``conformance/stories/build_fixture.py``."""
        fixture_path = Path(path)
        wire = FixtureWire.from_json(
            json.loads(fixture_path.read_text(encoding="utf-8")), str(fixture_path)
        )

        if not wire.embedder_id.endswith(f"@{wire.dimension}"):
            # Repeated at load time: a fixture can be hand-edited after generation.
            raise SimpleSemanticError(
                f"{fixture_path}: embedder_id {wire.embedder_id!r} disagrees with "
                f"dimension {wire.dimension}"
            )

        def vectors(section: str, records: list[FixtureRecordWire]) -> dict[str, np.ndarray]:
            out: dict[str, np.ndarray] = {}
            for record in records:
                if len(record.vector) != wire.dimension:
                    raise SimpleSemanticError(
                        f"{fixture_path}: {section} entry {record.id or record.key} has "
                        f"{len(record.vector)} dimensions, expected {wire.dimension}"
                    )
                out[record.key] = np.asarray(record.vector, dtype=np.float32)
            return out

        return cls(
            embedder_id=wire.embedder_id,
            dimension=wire.dimension,
            documents=vectors("documents", wire.documents),
            queries=vectors("queries", wire.queries),
            produces_normalized=wire.normalized,
            source=str(fixture_path),
        )

    def __len__(self) -> int:
        return len(self._documents) + len(self._queries)

    def _lookup(self, mode: str, table: dict[str, np.ndarray], text: str) -> np.ndarray:
        key = content_key(text)
        vector = table.get(key)
        if vector is None:
            raise ReplayMissError(mode, text, key, self._source)
        # Copy: a caller must not be able to mutate the recording.
        return vector.copy()

    async def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.stack([self._lookup("document", self._documents, text) for text in texts])

    async def embed_query(self, text: str) -> np.ndarray:
        # A different table from embed_documents: RETRIEVAL_QUERY here,
        # RETRIEVAL_DOCUMENT there, so the same string has two right answers.
        return self._lookup("query", self._queries, text)
