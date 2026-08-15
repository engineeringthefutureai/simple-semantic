"""The index: open, add, upsert, delete, search, compact.

``scores = D @ q`` then top-k. The rest is bookkeeping so that the matrix stays
correct across updates.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import canonical_json
from . import format as fmt
from .embedder import Embedder, normalize_row, normalize_rows
from .errors import CorruptIndexError, EmbedderMismatchError, SimpleSemanticError
from .wire import DocumentWire, decode

DEFAULT_CHUNKER_ID = "none"


@dataclass(frozen=True)
class Document:
    """One indexable unit. Already chunked — see :mod:`.chunker`."""

    id: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    id: str
    score: float
    text: str
    meta: dict[str, Any]
    row: int


@dataclass(frozen=True)
class AddResult:
    """What :meth:`SemanticIndex.add_all` did.

    ``skipped`` counts documents whose content hash already matched a live row,
    so no embedding call was made.
    """

    added: int
    replaced: int
    skipped: int


def _append_and_sync(path: Path, data: bytes | bytearray) -> None:
    """Append and make durable before the manifest that describes the new length."""
    with open(path, "ab") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


class SemanticIndex:
    """A directory of four files, plus the in-memory maps needed to serve it.

    Loaded at open: ids, metadata and content hashes. Not loaded: document text,
    fetched through ``offsets.bin`` on demand.
    """

    def __init__(self, path: Path, embedder: Embedder) -> None:
        # Use open() or create(); this constructor takes no responsibility for
        # validating the directory.
        self._path = path
        self._embedder = embedder
        self._manifest: fmt.Manifest
        self._ids: list[str] = []
        self._hashes: list[str] = []
        self._by_id: dict[str, int] = {}
        self._offsets: np.ndarray = np.zeros(1, dtype=fmt.OFFSET_DTYPE)
        self._tombstones = fmt.Tombstones(0)
        self._vectors: np.ndarray = np.zeros((0, embedder.dimension), dtype=fmt.VECTOR_DTYPE)
        self._closed = False

    # ---------------------------------------------------------------- lifecycle

    @classmethod
    def create(
        cls,
        path: str | Path,
        embedder: Embedder,
        *,
        chunker_id: str = DEFAULT_CHUNKER_ID,
        exist_ok: bool = False,
    ) -> SemanticIndex:
        """Create an empty index directory."""
        directory = Path(path)
        if (directory / fmt.MANIFEST).exists() and not exist_ok:
            raise SimpleSemanticError(
                f"{directory} already contains a {fmt.MANIFEST}; pass exist_ok=True to overwrite it"
            )
        directory.mkdir(parents=True, exist_ok=True)

        now = fmt.utc_now()
        manifest = fmt.Manifest(
            embedder_id=embedder.id,
            dimension=embedder.dimension,
            chunker_id=chunker_id,
            row_count=0,
            live_count=0,
            created_at=now,
            updated_at=now,
        )
        (directory / fmt.VECTORS).write_bytes(b"")
        (directory / fmt.DOCS).write_bytes(b"")
        (directory / fmt.OFFSETS).write_bytes(np.zeros(1, dtype=fmt.OFFSET_DTYPE).tobytes())
        fmt.write_atomic(directory / fmt.MANIFEST, manifest.to_bytes())

        return cls.open(directory, embedder)

    @classmethod
    def open(cls, path: str | Path, embedder: Embedder) -> SemanticIndex:
        """Open an existing index, running every check in SPEC.md §2.2."""
        directory = Path(path)
        manifest = fmt.read_manifest(directory)

        # SPEC.md §2.1. Do not relax this into a warning.
        if manifest.embedder_id != embedder.id:
            raise EmbedderMismatchError(str(directory), manifest.embedder_id, embedder.id)
        if manifest.dimension != embedder.dimension:
            raise CorruptIndexError(
                f"{directory}: manifest dimension {manifest.dimension} but embedder "
                f"{embedder.id!r} produces {embedder.dimension}"
            )

        index = cls(directory, embedder)
        index._manifest = manifest
        index._offsets = fmt.read_offsets(directory, manifest.row_count)
        index._tombstones = fmt.read_tombstones(directory, manifest.row_count)
        index._vectors = fmt.open_vectors(directory, manifest.row_count, manifest.dimension)
        index._load_docs()

        live = manifest.row_count - index._tombstones.deleted_count
        if live != manifest.live_count:
            raise CorruptIndexError(
                f"{directory}: manifest live_count {manifest.live_count} disagrees with "
                f"{fmt.TOMBSTONES} ({live} live of {manifest.row_count})"
            )
        return index

    def _load_docs(self) -> None:
        """Scan docs.jsonl once, building the id/meta/hash arrays and the id map.

        Ascending order with "latest live row wins": the previous row for an id
        was tombstoned before the new one was appended.
        """
        self._ids = []
        self._hashes = []
        self._by_id = {}
        docs_path = self._path / fmt.DOCS
        with open(docs_path, "rb") as handle:
            for row, raw in enumerate(handle):
                if row >= self._manifest.row_count:
                    raise CorruptIndexError(
                        f"{docs_path} has more than the {self._manifest.row_count} lines "
                        f"the manifest declares"
                    )
                document = self._decode_line(raw.rstrip(b"\n"), str(docs_path))
                self._ids.append(document.id)
                self._hashes.append(document.hash)
        if len(self._ids) != self._manifest.row_count:
            raise CorruptIndexError(
                f"{docs_path} has {len(self._ids)} lines but the manifest declares "
                f"row_count {self._manifest.row_count}"
            )
        for row, doc_id in enumerate(self._ids):
            if not self._tombstones.is_deleted(row):
                self._by_id[doc_id] = row

    def close(self) -> None:
        self.close_mapping()
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise SimpleSemanticError(f"{self._path}: index is closed")

    def __enter__(self) -> SemanticIndex:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------- inspection

    @property
    def path(self) -> Path:
        return self._path

    @property
    def manifest(self) -> fmt.Manifest:
        return self._manifest

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    def size(self) -> int:
        """Total rows, live and tombstoned."""
        return self._manifest.row_count

    def live_count(self) -> int:
        return self._manifest.live_count

    def contains(self, doc_id: str) -> bool:
        return doc_id in self._by_id

    def get(self, doc_id: str) -> Document | None:
        row = self._by_id.get(doc_id)
        if row is None:
            return None
        return self._document_at(row)

    def ids(self) -> list[str]:
        """Live ids, in row order."""
        return [self._ids[row] for row in sorted(self._by_id.values())]

    def stats(self) -> dict[str, Any]:
        vectors_bytes = self._manifest.row_count * self._manifest.dimension * 4
        return {
            "path": str(self._path),
            "embedder_id": self._manifest.embedder_id,
            "chunker_id": self._manifest.chunker_id,
            "dimension": self._manifest.dimension,
            "row_count": self._manifest.row_count,
            "live_count": self._manifest.live_count,
            "deleted_count": self._manifest.row_count - self._manifest.live_count,
            "vectors_bytes": vectors_bytes,
            "created_at": self._manifest.created_at,
            "updated_at": self._manifest.updated_at,
        }

    def _document_at(self, row: int) -> Document:
        """Fetch one document by row, via offsets.bin. SPEC.md §5."""
        start = int(self._offsets[row])
        end = int(self._offsets[row + 1])
        with open(self._path / fmt.DOCS, "rb") as handle:
            handle.seek(start)
            raw = handle.read(end - start)
        wire = self._decode_line(raw.rstrip(b"\n"), str(self._path / fmt.DOCS))
        return Document(id=wire.id, text=wire.text, meta=dict(wire.meta))

    @staticmethod
    def _decode_line(raw: bytes, source: str) -> DocumentWire:
        try:
            return decode(DocumentWire, canonical_json.decode_line(raw), source)
        except SimpleSemanticError as exc:
            raise CorruptIndexError(str(exc)) from exc
        except ValueError as exc:
            raise CorruptIndexError(f"{source}: invalid document line ({exc})") from exc

    # ------------------------------------------------------------------ writes

    async def add_all(self, documents: Iterable[Document]) -> AddResult:
        """Add or replace documents. Unchanged content is not re-embedded.

        Keyed on the content hash from SPEC.md §4.1, which folds in the embedder
        and chunker ids.
        """
        docs = list(documents)
        if not docs:
            return AddResult(0, 0, 0)

        pending: list[tuple[Document, str]] = []
        skipped = 0
        # Track hashes staged in this call so a repeat inside one batch is also
        # a no-op rather than an append of an identical row.
        staged: dict[str, str] = {}
        for doc in docs:
            if not doc.id:
                raise SimpleSemanticError("document id must be a non-empty string")
            digest = fmt.content_hash(doc.text, self._embedder.id, self._manifest.chunker_id)
            if doc.id in staged:
                if staged[doc.id] == digest:
                    skipped += 1
                    continue
            elif doc.id in self._by_id and self._hashes[self._by_id[doc.id]] == digest:
                skipped += 1
                continue
            staged[doc.id] = digest
            pending.append((doc, digest))

        if not pending:
            return AddResult(0, 0, skipped)

        # Validate before embedding: an embedding call costs money.
        for doc, _ in pending:
            canonical_json.validate_meta(doc.meta)

        vectors = await self._embed_documents([doc.text for doc, _ in pending])
        # Normalized at write time whatever the embedder claims. SPEC.md §3.1.
        vectors = normalize_rows(vectors)

        replaced = 0
        vector_bytes = bytearray()
        doc_bytes = bytearray()
        new_offsets: list[int] = []
        next_offset = int(self._offsets[-1])
        first_new_row = self._manifest.row_count
        # Grow the bitmap up front: an id repeated inside this same batch
        # tombstones a row that only exists because of an earlier iteration.
        self._tombstones.grow_to(first_new_row + len(pending))

        for i, (doc, digest) in enumerate(pending):
            previous = self._by_id.get(doc.id)
            if previous is not None:
                self._tombstones.mark_deleted(previous)
                replaced += 1
            row = first_new_row + i
            vector_bytes += vectors[i].astype(fmt.VECTOR_DTYPE).tobytes()
            line = canonical_json.encode_document(doc.id, doc.text, doc.meta, digest) + b"\n"
            doc_bytes += line
            next_offset += len(line)
            new_offsets.append(next_offset)
            self._ids.append(doc.id)
            self._hashes.append(digest)
            self._by_id[doc.id] = row

        self._append(vector_bytes, doc_bytes, new_offsets)
        return AddResult(added=len(pending) - replaced, replaced=replaced, skipped=skipped)

    async def upsert(self, document: Document) -> AddResult:
        """Add one document, replacing any live row with the same id."""
        return await self.add_all([document])

    async def _embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed in batches the embedder declares it can take.

        Batching lives here so no code path can call the embedder once per
        document in a loop over results.
        """
        limit = max(1, self._embedder.max_batch_size)
        blocks: list[np.ndarray] = []
        for start in range(0, len(texts), limit):
            block = await self._embedder.embed_documents(texts[start : start + limit])
            array = np.asarray(block, dtype=np.float32)
            if array.ndim != 2 or array.shape[1] != self._manifest.dimension:
                raise SimpleSemanticError(
                    f"embedder {self._embedder.id!r} returned shape {array.shape}, "
                    f"expected (n, {self._manifest.dimension})"
                )
            if array.shape[0] != len(texts[start : start + limit]):
                raise SimpleSemanticError(
                    f"embedder {self._embedder.id!r} returned {array.shape[0]} vectors "
                    f"for {len(texts[start : start + limit])} texts"
                )
            blocks.append(array)
        return np.concatenate(blocks) if blocks else np.zeros((0, self._manifest.dimension), "f4")

    def _append(
        self,
        vector_bytes: bytes | bytearray,
        doc_bytes: bytes | bytearray,
        new_offsets: Sequence[int],
    ) -> None:
        """Append to all four files, then commit by rewriting the manifest.

        The manifest is written last because it declares how long the others
        should be: a crash before it leaves a vectors.f32 longer than row_count
        implies, which SPEC.md §2.2's length check catches on the next open.

        The appends are fsynced first, so the reverse — a durable manifest
        counting rows that never reached the disk — cannot happen either. That
        one would refuse the whole index rather than lose the last batch.
        """
        self.close_mapping()
        _append_and_sync(self._path / fmt.VECTORS, vector_bytes)
        _append_and_sync(self._path / fmt.DOCS, doc_bytes)

        self._offsets = np.concatenate(
            [self._offsets, np.array(new_offsets, dtype=fmt.OFFSET_DTYPE)]
        )
        row_count = len(self._ids)
        self._tombstones.grow_to(row_count)

        fmt.write_atomic(self._path / fmt.OFFSETS, self._offsets.tobytes())
        self._write_tombstones()

        self._manifest.row_count = row_count
        self._manifest.live_count = row_count - self._tombstones.deleted_count
        self._manifest.updated_at = fmt.utc_now()
        fmt.write_atomic(self._path / fmt.MANIFEST, self._manifest.to_bytes())

        self._vectors = fmt.open_vectors(self._path, row_count, self._manifest.dimension)

    def _write_tombstones(self) -> None:
        path = self._path / fmt.TOMBSTONES
        if self._tombstones.any_deleted():
            fmt.write_atomic(path, self._tombstones.to_bytes())
        elif path.exists():
            path.unlink()

    def close_mapping(self) -> None:
        """Drop the memmap before the underlying file changes length."""
        vectors = self._vectors
        if isinstance(vectors, np.memmap):
            vectors._mmap.close()  # type: ignore[attr-defined]
        self._vectors = np.zeros((0, self._manifest.dimension), dtype=fmt.VECTOR_DTYPE)

    def delete(self, doc_id: str) -> bool:
        """Tombstone the live row for ``doc_id``. Unknown ids are a no-op, not an error."""
        row = self._by_id.pop(doc_id, None)
        if row is None:
            return False
        self._tombstones.mark_deleted(row)
        self._write_tombstones()
        self._manifest.live_count = self._manifest.row_count - self._tombstones.deleted_count
        self._manifest.updated_at = fmt.utc_now()
        fmt.write_atomic(self._path / fmt.MANIFEST, self._manifest.to_bytes())
        return True

    def compact(self) -> int:
        """Rewrite the index without tombstoned rows. Returns rows dropped.

        Crash-safe: the new index is built complete in a sibling directory and
        fsynced before anything in place is touched.
        """
        dropped = self._manifest.row_count - self._manifest.live_count
        if dropped == 0:
            return 0

        live_rows = np.flatnonzero(self._tombstones.live_mask()).tolist()
        staging = self._path.parent / f"{self._path.name}.compact.tmp"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)

        offsets: list[int] = [0]
        cursor = 0
        with (
            open(staging / fmt.VECTORS, "wb") as vec_out,
            open(staging / fmt.DOCS, "wb") as doc_out,
            open(self._path / fmt.DOCS, "rb") as doc_in,
        ):
            for row in live_rows:
                vec_out.write(self._vectors[row].astype(fmt.VECTOR_DTYPE).tobytes())
                start = int(self._offsets[row])
                end = int(self._offsets[row + 1])
                doc_in.seek(start)
                line = doc_in.read(end - start)
                doc_out.write(line)
                cursor += len(line)
                offsets.append(cursor)
            vec_out.flush()
            doc_out.flush()
            os.fsync(vec_out.fileno())
            os.fsync(doc_out.fileno())

        fmt.write_atomic(staging / fmt.OFFSETS, np.array(offsets, dtype=fmt.OFFSET_DTYPE).tobytes())
        compacted = fmt.Manifest(
            embedder_id=self._manifest.embedder_id,
            dimension=self._manifest.dimension,
            chunker_id=self._manifest.chunker_id,
            row_count=len(live_rows),
            live_count=len(live_rows),
            created_at=self._manifest.created_at,
            updated_at=fmt.utc_now(),
        )
        fmt.write_atomic(staging / fmt.MANIFEST, compacted.to_bytes())
        fmt.fsync_path(staging)

        self.close_mapping()
        retired = self._path.parent / f"{self._path.name}.compact.old"
        if retired.exists():
            shutil.rmtree(retired)
        self._path.rename(retired)
        staging.rename(self._path)
        fmt.fsync_path(self._path.parent)
        shutil.rmtree(retired)

        self._manifest = fmt.read_manifest(self._path)
        self._offsets = fmt.read_offsets(self._path, self._manifest.row_count)
        self._tombstones = fmt.read_tombstones(self._path, self._manifest.row_count)
        self._vectors = fmt.open_vectors(
            self._path, self._manifest.row_count, self._manifest.dimension
        )
        self._load_docs()
        return dropped

    # ------------------------------------------------------------------ search

    async def search(self, query: str, k: int = 10) -> list[SearchResult]:
        """Embed the query, then run exact k-NN over the live rows."""
        if not query.strip():
            # No direction; better than ranking against the e_0 fallback.
            return []
        return self.search_vector(await self._embedder.embed_query(query), k=k)

    def search_vector(self, query_vector: np.ndarray, k: int = 10) -> list[SearchResult]:
        """Exact k-NN against a pre-computed query vector.

        Public so a caller with a cached embedding or a centroid need not go
        back through the embedder.
        """
        self._require_open()
        if k <= 0:
            return []
        if self._manifest.row_count == 0:
            return []

        query = normalize_row(np.asarray(query_vector, dtype=np.float32))
        if query.shape != (self._manifest.dimension,):
            raise SimpleSemanticError(
                f"query vector has shape {query.shape}, expected ({self._manifest.dimension},)"
            )

        rows = np.flatnonzero(self._tombstones.live_mask())
        if rows.size == 0:
            return []

        # float64: a float32 dot loses enough at 3072 dimensions to reorder near-ties.
        if rows.size == self._manifest.row_count:
            matrix = np.asarray(self._vectors, dtype=np.float64)
        else:
            matrix = self._vectors[rows].astype(np.float64)
        scores = matrix @ query.astype(np.float64)

        limit = min(k, rows.size)
        if limit < rows.size:
            # O(n) selection, no full sort. argpartition alone keeps an
            # arbitrary subset of the rows tied on the k-th score, so take
            # everything strictly better and fill from the tied rows in row
            # order. SPEC.md §8. flatnonzero returns ascending indices and
            # `rows` is ascending, so "first tied" is "lowest row".
            threshold = scores[np.argpartition(-scores, limit - 1)[limit - 1]]
            better = np.flatnonzero(scores > threshold)
            tied = np.flatnonzero(scores == threshold)
            candidates = np.concatenate([better, tied[: limit - better.size]])
        else:
            candidates = np.arange(rows.size)

        selected_scores = scores[candidates]
        selected_rows = rows[candidates]
        # Ties break by ascending row index. SPEC.md §8.
        order = np.lexsort((selected_rows, -selected_scores))

        results: list[SearchResult] = []
        for position in order.tolist():
            row = int(selected_rows[position])
            document = self._document_at(row)
            results.append(
                SearchResult(
                    id=document.id,
                    score=float(selected_scores[position]),
                    text=document.text,
                    meta=document.meta,
                    row=row,
                )
            )
        return results
