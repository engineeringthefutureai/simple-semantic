"""``ReplayEmbedder`` — serve recorded vectors instead of calling a model.

The third embedder, and the one that closes a real gap. ``HashingEmbedder`` is
deterministic but semantically meaningless: it can prove the *format* is
correct, and nothing about whether search actually retrieves. ``GeminiEmbedder``
retrieves properly but needs a credential, a network, and money, so no test
suite can depend on it.

``ReplayEmbedder`` reads vectors a real model produced once, recorded to a JSON
fixture. That makes genuine retrieval-quality assertions — top-1 accuracy,
negative controls ranking below real matches — runnable in CI, offline, with no
key and byte-identical results every run.

A miss is a loud error, never a zero vector or a fallback. A mock that silently
invents a plausible answer is the failure mode this whole project is built to
refuse: it would turn "your text is not in the recording" into "your retrieval
quality quietly got worse".
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .errors import SimpleSemanticError


class ReplayMissError(SimpleSemanticError):
    """The text asked for is not in the recording.

    Names the mode as well as the text, because the commonest cause is asking
    for a document embedding of something recorded only as a query. Those are
    genuinely different vectors — different task types — and serving one for the
    other is a silent quality loss.
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

    Reports ``produces_normalized`` honestly from the fixture. The story fixture
    says ``False``, because ``gemini-embedding-001`` pre-normalizes only its
    default 3072-dimension output and these vectors are 768 — which makes this
    embedder the one that exercises SPEC.md §3.1 write-time normalization
    against genuinely unnormalized input.
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
        raw: dict[str, Any] = json.loads(fixture_path.read_text(encoding="utf-8"))

        for key in ("embedder_id", "dimension", "documents", "queries"):
            if key not in raw:
                raise SimpleSemanticError(f"{fixture_path}: fixture is missing {key!r}")

        dimension = int(raw["dimension"])
        embedder_id = str(raw["embedder_id"])
        if not embedder_id.endswith(f"@{dimension}"):
            # The same guard build_fixture.py applies, repeated at load time
            # because a fixture can be hand-edited after it is generated.
            raise SimpleSemanticError(
                f"{fixture_path}: embedder_id {embedder_id!r} disagrees with dimension {dimension}"
            )

        def load(section: str) -> dict[str, np.ndarray]:
            out: dict[str, np.ndarray] = {}
            for record in raw[section]:
                vector = np.asarray(record["vector"], dtype=np.float32)
                if vector.shape != (dimension,):
                    raise SimpleSemanticError(
                        f"{fixture_path}: {section} entry {record.get('id', record['key'])} "
                        f"has {vector.shape[0]} dimensions, expected {dimension}"
                    )
                out[str(record["key"])] = vector
            return out

        return cls(
            embedder_id=embedder_id,
            dimension=dimension,
            documents=load("documents"),
            queries=load("queries"),
            produces_normalized=bool(raw.get("normalized", False)),
            source=str(fixture_path),
        )

    def __len__(self) -> int:
        return len(self._documents) + len(self._queries)

    def _lookup(self, mode: str, table: dict[str, np.ndarray], text: str) -> np.ndarray:
        key = content_key(text)
        vector = table.get(key)
        if vector is None:
            raise ReplayMissError(mode, text, key, self._source)
        # Copy: a caller mutating the result must not corrupt the recording.
        return vector.copy()

    async def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.stack([self._lookup("document", self._documents, text) for text in texts])

    async def embed_query(self, text: str) -> np.ndarray:
        # Deliberately a different table from embed_documents. The recording was
        # made with RETRIEVAL_QUERY here and RETRIEVAL_DOCUMENT there, so the
        # same string has two different correct answers.
        return self._lookup("query", self._queries, text)
