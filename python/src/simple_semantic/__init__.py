"""simple-semantic: brute-force semantic search over a memory-mapped matrix.

Exact k-NN, 100% recall by construction, exact metadata filtering, trivial
updates. No ANN index, no vector database, no hidden machinery.

The on-disk format is specified in SPEC.md and is the primary artifact; this
package is one of two implementations of it.

``GeminiEmbedder`` lives in :mod:`simple_semantic.gemini` and is not imported
here, so that using the library does not require an HTTP client.
"""

from .canonical_json import encode as canonical_encode
from .chunker import Chunk, FixedChunker
from .embedder import Embedder, HashingEmbedder, normalize_row, normalize_rows, tokenize
from .errors import (
    CorruptIndexError,
    EmbedderMismatchError,
    ExtractionError,
    FormatVersionError,
    MetaValueError,
    SimpleSemanticError,
)
from .extract import SemanticId, SemanticIndexed, SemanticMeta, from_dataclass, from_lambdas
from .format import FORMAT_VERSION, Manifest, content_hash
from .index import AddResult, Document, Filter, SearchResult, SemanticIndex
from .replay import ReplayEmbedder, ReplayMissError

__all__ = [
    "FORMAT_VERSION",
    "AddResult",
    "Chunk",
    "CorruptIndexError",
    "Document",
    "Embedder",
    "EmbedderMismatchError",
    "ExtractionError",
    "Filter",
    "FixedChunker",
    "FormatVersionError",
    "HashingEmbedder",
    "Manifest",
    "MetaValueError",
    "ReplayEmbedder",
    "ReplayMissError",
    "SearchResult",
    "SemanticId",
    "SemanticIndex",
    "SemanticIndexed",
    "SemanticMeta",
    "SimpleSemanticError",
    "canonical_encode",
    "content_hash",
    "from_dataclass",
    "from_lambdas",
    "normalize_row",
    "normalize_rows",
    "tokenize",
]
