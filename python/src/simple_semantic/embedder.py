"""The ``Embedder`` boundary and the deterministic hashing embedder."""

from __future__ import annotations

import hashlib
import re
from typing import Protocol, runtime_checkable

import numpy as np

# Unicode letters and numbers only: [^\W_] is the equivalent of Java's
# \p{L}\p{N}, because Python's \w is "alphanumeric per str.isalnum(), plus
# underscore" and str.isalnum() covers exactly the L* and N* categories.
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


@runtime_checkable
class Embedder(Protocol):
    """Both implementations expose this shape.

    ``embed_documents`` and ``embed_query`` stay separate: Gemini needs
    ``taskType: RETRIEVAL_DOCUMENT`` versus ``RETRIEVAL_QUERY``, e5/BGE-style
    models need ``passage: `` / ``query: `` prefixes. A single ``embed()``
    makes that asymmetry unrepresentable.
    """

    id: str
    dimension: int
    max_batch_size: int
    produces_normalized: bool

    async def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    async def embed_query(self, text: str) -> np.ndarray: ...


def tokenize(text: str) -> list[str]:
    """Lowercase, Unicode-aware tokenization. SPEC.md appendix A.

    ``str.lower()`` is locale-independent and applies the same SpecialCasing
    rules as the JVM's ``lowercase(Locale.ROOT)``, so the two implementations
    agree.
    """
    tokens: list[str] = _TOKEN_RE.findall(text.lower())
    return tokens


def normalize_row(vector: np.ndarray) -> np.ndarray:
    """L2-normalize one row exactly as SPEC.md §3.1 specifies.

    ``np.cumsum`` rather than ``np.sum`` or ``np.dot``: cumsum is a specified
    sequential prefix scan, the others may use pairwise or blocked summation.
    One ulp of difference in float64 changes the final float32 rounding.
    """
    v64 = np.asarray(vector, dtype=np.float64)
    squares = v64 * v64
    ss = float(np.cumsum(squares)[-1]) if squares.size else 0.0
    norm = np.sqrt(ss)
    if norm == 0.0:
        # 0/0 would put a NaN in the matrix that poisons every later dot product.
        out = np.zeros(v64.shape[0], dtype=np.float32)
        if out.size:
            out[0] = np.float32(1.0)
        return out
    normalized: np.ndarray = (v64 / norm).astype(np.float32)
    return normalized


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Row-wise :func:`normalize_row` over a 2-D array.

    Batched because the cumsum needs a float64 scratch copy, and a full-matrix
    copy at 1M x 768 is 6 GB.
    """
    m = np.asarray(matrix)
    if m.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {m.shape}")
    out = np.empty(m.shape, dtype=np.float32)
    batch = max(1, 4096)
    for start in range(0, m.shape[0], batch):
        stop = min(start + batch, m.shape[0])
        chunk = m[start:stop].astype(np.float64)
        ss = np.cumsum(chunk * chunk, axis=1)[:, -1]
        norms = np.sqrt(ss)
        zero = norms == 0.0
        safe = np.where(zero, 1.0, norms)
        block = (chunk / safe[:, None]).astype(np.float32)
        if zero.any():
            block[zero] = 0.0
            block[zero, 0] = np.float32(1.0)
        out[start:stop] = block
    return out


class HashingEmbedder:
    """Deterministic, seeded, no network. SPEC.md appendix A.

    A signed hashing trick, not a good embedding — it exists to be identical
    in both languages bit for bit, which real models are not, and to make the
    test suite runnable with no API key.

    Documents and queries share the embedding function here so that search over
    a hashed index still returns sensible neighbours.
    """

    def __init__(self, dimension: int = 256, seed: int = 0) -> None:
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        self.dimension = dimension
        self.seed = seed
        self.id = f"hashing-{seed}@{dimension}"
        self.max_batch_size = 4096
        self.produces_normalized = True

    def embed_text(self, text: str) -> np.ndarray:
        """Synchronous single-text embedding. No I/O, so nothing to await."""
        acc = np.zeros(self.dimension, dtype=np.float64)
        seed_bytes = self.seed.to_bytes(8, "little", signed=False)
        for token in tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8") + b"\x00" + seed_bytes).digest()
            column = int.from_bytes(digest[0:4], "little", signed=False) % self.dimension
            sign = 1.0 if (digest[4] & 1) == 0 else -1.0
            acc[column] += sign
        # Counts are small integers, exact in both float64 and float32.
        return normalize_row(acc.astype(np.float32))

    async def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.stack([self.embed_text(t) for t in texts])

    async def embed_query(self, text: str) -> np.ndarray:
        return self.embed_text(text)


__all__ = [
    "Embedder",
    "HashingEmbedder",
    "normalize_row",
    "normalize_rows",
    "tokenize",
]
