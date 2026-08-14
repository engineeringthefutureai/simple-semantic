"""The ``Embedder`` boundary and the deterministic hashing embedder.

The embedder is the one part of this project that is not neanderthal — a
hosted model is a network call to someone else's GPU. Isolating it behind a
narrow interface is what makes the rest of the system honest, and what makes
the whole test suite runnable without an API key.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol, runtime_checkable

import numpy as np

# Unicode letters and numbers only. Python's ``\w`` is "alphanumeric per
# str.isalnum(), plus underscore", and str.isalnum() covers exactly the L* and
# N* general categories — so [^\W_] is the Unicode-correct equivalent of Java's
# \p{L}\p{N}. An ASCII-only [a-z0-9]+ would silently drop every non-Latin
# script, which is the kind of bug that only shows up in someone else's corpus.
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


@runtime_checkable
class Embedder(Protocol):
    """Both implementations expose this same shape. See CLAUDE.md.

    ``embed_documents`` and ``embed_query`` are separate methods and must stay
    separate. Gemini needs ``taskType: RETRIEVAL_DOCUMENT`` versus
    ``RETRIEVAL_QUERY``; e5/BGE-style models need ``passage: `` / ``query: ``
    prefixes. A single ``embed()`` makes that asymmetry unrepresentable, and
    getting it wrong costs retrieval quality without raising anything.
    """

    id: str
    dimension: int
    max_batch_size: int
    produces_normalized: bool

    async def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    async def embed_query(self, text: str) -> np.ndarray: ...


def tokenize(text: str) -> list[str]:
    """Lowercase, Unicode-aware tokenization.

    ``str.lower()`` is locale-independent in Python and applies the same
    SpecialCasing rules as the JVM's ``lowercase(Locale.ROOT)`` — including
    Greek final sigma and the dotted capital I — so the two implementations
    agree. Conformance covers Cyrillic, accented Latin and CJK to keep that
    honest.
    """
    tokens: list[str] = _TOKEN_RE.findall(text.lower())
    return tokens


def normalize_row(vector: np.ndarray) -> np.ndarray:
    """L2-normalize one row exactly as SPEC.md §3.1 specifies.

    Sequential float64 accumulation, ascending index. ``np.cumsum`` is used
    rather than ``np.sum`` or ``np.dot`` because cumsum is specified as a
    sequential prefix scan, while the other two are free to use pairwise or
    blocked summation. That difference is one ulp in float64, which is enough
    to change the final float32 rounding and break byte-identity with the
    Kotlin writer.
    """
    v64 = np.asarray(vector, dtype=np.float64)
    squares = v64 * v64
    ss = float(np.cumsum(squares)[-1]) if squares.size else 0.0
    norm = np.sqrt(ss)
    if norm == 0.0:
        # A zero row has no direction, and 0/0 would put a NaN in the matrix
        # that silently poisons every later dot product. e_0 is arbitrary but
        # defined: a valid unit vector that simply ranks poorly.
        out = np.zeros(v64.shape[0], dtype=np.float32)
        if out.size:
            out[0] = np.float32(1.0)
        return out
    normalized: np.ndarray = (v64 / norm).astype(np.float32)
    return normalized


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Row-wise :func:`normalize_row` over a 2-D array, in row-batches.

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

    Required, not optional: it is what makes the entire behavioural suite and
    the cross-implementation conformance job runnable with no API key and no
    network. A test suite that needs a credential is a test suite that stops
    being run.

    The signed hashing trick — one +/-1 per token into a hashed column — is
    not a good embedding. It is not meant to be. It is meant to be *identical*
    in both languages, bit for bit, which real models are not.

    Documents and queries share the embedding function here, so that search
    over a hashed index still returns sensible neighbours. The interface's
    document/query split is exercised by :class:`GeminiEmbedder`, where the
    asymmetry is real.
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
        """Synchronous single-text embedding. No I/O, so no reason to await it."""
        acc = np.zeros(self.dimension, dtype=np.float64)
        seed_bytes = self.seed.to_bytes(8, "little", signed=False)
        for token in tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8") + b"\x00" + seed_bytes).digest()
            column = int.from_bytes(digest[0:4], "little", signed=False) % self.dimension
            sign = 1.0 if (digest[4] & 1) == 0 else -1.0
            acc[column] += sign
        # Counts are small integers, exact in both float64 and float32, so this
        # cast cannot introduce a cross-language difference.
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
