> **⚠️ THIS FILE IS 100% AI-GENERATED.**
>
> Every word of this document was written by an AI model, not by a human.
> It has not been human-authored, and its claims, design decisions, and
> technical assertions have not been independently verified by a person.
> Treat it as machine-generated project guidance: useful as a starting
> point, not as reviewed engineering documentation.

# simple-semantic

Brute-force semantic search. No ANN index, no vector database, no hidden
machinery. A sibling to
[`simple-fts`](https://github.com/engineeringthefutureai/simple-fts), which
does the same thing for inverted-index full-text search.

Two implementations — Kotlin and Python — over **one shared on-disk format**.
The format is the primary artifact. The implementations exist to prove it is
real: an index written by either must be readable by the other, byte-for-byte,
verified in CI.

GitHub description line:

> Brute-force semantic search index with one on-disk format and two
> implementations (Kotlin, Python)

Topics: `semantic-search`, `embeddings`, `vector-search`, `knn`, `kotlin`,
`python`. Cross-link `simple-fts` from this repo's README and this repo from
`simple-fts`'s description, so the pairing is discoverable from either side.

---

## Thesis

Exact k-NN over a dense matrix is `scores = D @ q` followed by a top-k
selection. That is the whole algorithm. Everything a vector database adds is
either an approximation of this or a preprocessing step around it.

At the scale most people actually operate — 10k to 1M chunks — brute force is
not merely "good enough". It is **strictly better on three axes**, and the
README must make this argument with benchmarks, not assertions:

1. **Filtered search is exact.** Metadata predicates become a boolean mask.
   The filter *reduces* work. HNSW must choose between pre-filtering (which
   breaks graph connectivity and strands traversal in dead ends) and
   post-filtering (fetch top-1000, hope enough survive). Neither degrades
   gracefully as selectivity rises.
2. **Updates and deletes are trivial.** Tombstone a row, append a row. No
   graph repair, no index rebuild.
3. **Recall is 100% by construction.** ANN trades recall for latency. That is
   a real cost, and it is absent from most marketing.

The honest limit: the hosted embedding model is the one thing that is not
neanderthal. Isolate it behind an interface, and later ship a local
implementation so the gap can be measured rather than asserted.

---

## Non-goals

Do not add these. If a task seems to require one, stop and ask.

- ANN indexing (HNSW, IVF, LSH, ScaNN, product quantization)
- A dependency on Chroma, FAISS, Qdrant, Milvus, pgvector, or LanceDB
- A vendored BLAS in the Kotlin implementation — the hand-written dot loop
  *is* the point there
- Distributed anything, sharding, or a network protocol between the two
  implementations
- LLM generation, agents, or a full RAG chain. This is retrieval only.
- Hiding an embedding network call behind a signature that looks synchronous

---

## Repository layout

```
simple-semantic/
  SPEC.md              # the on-disk format, normative. Version it.
  README.md            # thesis, benchmarks, usage in both languages
  LICENSE              # Apache-2.0. Commit this first.
  kotlin/
    core/              # library
    cli/               # index / search / stats / compact
    build.gradle.kts
  python/
    src/simple_semantic/
    tests/
    pyproject.toml
  conformance/
    fixtures/          # golden indexes, committed, small (< 1 MB)
    corpus/            # shared test corpus, committed
    run.sh             # cross-implementation verification
  .github/workflows/
```

---

## On-disk format (normative — mirror into SPEC.md)

An index is a **directory**, not a single file. Four files. Every one is
inspectable with standard tools; that is a design requirement, not a
nicety.

### `manifest.json`

```json
{
  "format_version": 1,
  "embedder_id": "gemini-embedding-001@768",
  "dimension": 768,
  "normalized": true,
  "chunker_id": "fixed-512-overlap-64",
  "hash_algorithm": "sha256",
  "row_count": 12043,
  "live_count": 11890,
  "created_at": "2026-08-13T17:00:00Z",
  "updated_at": "2026-08-13T19:22:41Z"
}
```

`embedder_id` is load-bearing. **`open()` MUST refuse to serve an index whose
manifest `embedder_id` differs from the configured embedder**, with an error
naming both. This is the single most important correctness rule in the
project. A model swap against an existing index produces no exception and no
crash — just silently meaningless rankings. Most vector databases do not guard
this. We do, and the README says so.

On open, also verify `filesize(vectors.f32) == row_count * dimension * 4` and
fail loudly on mismatch.

### `vectors.f32`

Flat, row-major, **little-endian IEEE-754 binary32**. No header — the header
is `manifest.json`. Length is exactly `row_count * dimension * 4` bytes.

The absence of a header is deliberate: it makes the file directly mappable
with zero parsing on both platforms.

- Python: `np.memmap(path, dtype="<f4", mode="r", shape=(row_count, dim))`
- Kotlin: `FileChannel.map(READ_ONLY, 0, size, arena)` → `MemorySegment`

**Use the FFM API (`MemorySegment`, `Arena`), not `ByteBuffer`.** ByteBuffer
caps at 2 GB, which at 3072 dimensions is ~175k rows — a limit a real
knowledge base will hit, forcing a redesign. FFM maps past it.

**Endianness footgun:** JVM `ByteBuffer` defaults to big-endian. Any code path
touching bytes must set `ByteOrder.LITTLE_ENDIAN` explicitly. Add a
conformance test that would fail if this regressed.

All rows MUST be L2-normalized at write time, regardless of what the embedder
claims. Normalization is idempotent, costs one pass, and makes the index
immune to embedder changes. Assert `abs(norm - 1.0) < 1e-5` on every row in a
test.

### `docs.jsonl`

UTF-8, one JSON object per line, LF-terminated. **Line N corresponds to row N
of `vectors.f32`.** That correspondence is the core invariant of the format.

```json
{"id": "notes/embeddings.md#3", "text": "...", "meta": {"source": "notes/embeddings.md", "tags": ["ml"]}, "hash": "9f86d0..."}
```

`hash` is `sha256(text + "\0" + embedder_id + "\0" + chunker_id)`, hex-encoded.
It drives incremental indexing: on re-index, a chunk whose hash is unchanged is
not re-embedded. This is the feature that determines whether the tool is usable
daily, and it is the API cost control.

### `offsets.bin`

`row_count + 1` unsigned 64-bit little-endian byte offsets into `docs.jsonl`.
Entry `i` is the start of line `i`; entry `row_count` is the file length.
Enables O(1) random access to a document without parsing the whole file.

### `tombstones.bits`

`ceil(row_count / 8)` bytes. Bit `i` (LSB-first within each byte) set means
row `i` is deleted. Absent file means no deletions.

---

## Update semantics

**Append-only. Never mutate a row in place.**

`upsert(doc)`: if the id already has a live row, set its tombstone bit, then
append the new vector and document as a new row. The in-memory `id → row` map
(rebuilt from `docs.jsonl` on load, latest live row wins) points at the new
row.

This is the direct correction of the defect found in `simple-fts`, where the
inverted index stored whole document objects and an update refreshed only the
postings reachable through *newly added* terms — leaving stale copies of the
same logical document alive under shared terms. Here it is structurally
impossible: the vector matrix holds no document, only a row, and the row index
*is* the identity.

Write tests that would catch the `simple-fts` bug class if it were possible:

- Upsert a document, then search a term present in both versions. Assert the
  returned text is the new version.
- Assert result ids are distinct across a query matching many rows.
- Assert a tombstoned row never appears, even when it is the nearest vector.
- Assert `live_count` after N upserts of the same id is 1.

`compact()` rewrites all four files dropping tombstoned rows. It must be
crash-safe: write to a sibling temp directory, fsync, then atomic rename.

---

## The `Embedder` interface

The one boundary that matters. Both implementations expose the same shape.

```kotlin
interface Embedder {
    val id: String                    // "gemini-embedding-001@768"
    val dimension: Int
    val maxBatchSize: Int
    val producesNormalized: Boolean

    suspend fun embedDocuments(texts: List<String>): List<FloatArray>
    suspend fun embedQuery(text: String): FloatArray
}
```

```python
class Embedder(Protocol):
    id: str
    dimension: int
    max_batch_size: int
    produces_normalized: bool

    async def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    async def embed_query(self, text: str) -> np.ndarray: ...
```

Rules:

- **Document and query embedding are separate methods and must stay separate.**
  Gemini needs `taskType: RETRIEVAL_DOCUMENT` vs `RETRIEVAL_QUERY`; e5/BGE-style
  models need `passage: ` / `query: ` prefixes. A single `embed()` makes the
  asymmetry unrepresentable, and getting it wrong is a silent quality loss.
- Batching, retry, and rate-limit handling belong to the implementation, not
  the index.
- The index never calls an embedder in a loop over results.

Ship three implementations:

1. `GeminiEmbedder` — `gemini-embedding-001`. Note in code that **only the
   default 3072-dim output is pre-normalized**; any `output_dimensionality`
   below that returns unnormalized vectors and must be normalized manually.
   `gemini-embedding-2` normalizes truncated output, but aggregates multiple
   inputs in one request into a single embedding unless each is wrapped
   individually — that difference will silently corrupt a batch loop written
   against `001`. Default to 768 dimensions.
2. `HashingEmbedder` — deterministic, seeded, no network. Makes the entire test
   suite runnable without an API key. Required, not optional.
3. `LocalEmbedder` — ONNX, small bi-encoder, CPU. Python first
   (`onnxruntime` + `tokenizers`); Kotlin via ONNX Runtime's Java bindings
   afterwards. This is what makes the benchmark post possible.

---

## Search

```
mask   = live_rows AND metadata_predicate
scores = vectors[mask] @ query
top_k  = bounded_min_heap(scores, k)      # O(n log k), never a full sort
```

Requirements:

- Metadata filtering is a boolean mask applied before the dot products. Exact,
  and cheaper as the filter gets more selective. Benchmark this against an ANN
  library and put the curve in the README — it is the strongest single result
  the project can produce.
- Top-k via a bounded heap. Do not sort all N.
- **Never expose or hardcode an absolute score threshold.** Embedding spaces
  are anisotropic; cosine similarities cluster in a narrow band that shifts
  with model and corpus. Only relative ordering carries signal. If a threshold
  API is requested, push back.
- Kotlin: plain `FloatArray` dot loop. C2 auto-vectorizes it acceptably. Do
  **not** reach for the Vector API — it is still incubating and would force
  `--add-modules` on every consumer for a constant factor. Revisit only if a
  JMH benchmark justifies it.
- Python: `D @ q` via numpy. Note honestly in the README that this delegates to
  BLAS and is therefore the least neanderthal line in the project.

Benchmark and publish: exact k-NN over a matrix-vector product is
**memory-bandwidth-bound, not compute-bound**. 1M × 768 fp32 is ~3 GB; the
arithmetic is trivial next to the streaming cost. The optimization lever is
therefore dtype, not cleverness — measure fp32 vs fp16 vs int8-with-fp32-rerank
and show the curve. This result contradicts the common intuition that an ANN
index is required, which is exactly why it belongs in the README.

---

## Public API

Both implementations expose annotation/decorator-driven extraction for the
demo, mirroring `simple-fts`:

```kotlin
data class Note(
    @SemanticId val id: String,
    @SemanticIndexed val title: String,
    @SemanticIndexed val body: String?,
)
```

**And a public factory taking plain lambdas.** In `simple-fts` the
lambda-based constructor existed but was `internal`, with the only public
factory hard-wiring the annotation extractors — the extension point was built
and then sealed off. Do not repeat that. Both entry points are public here.

Null-safe extraction: an annotated field declared nullable must not throw. The
`simple-fts` extractor cast with `value as String` and blew up on exactly the
nullable field its own README example declared.

Tokenization/normalization of any text handled by the library uses
`\p{L}\p{N}` classes and `Locale.ROOT` casing. ASCII-only regexes silently drop
non-Latin text.

Also provide: `size()`, `liveCount()`, `contains(id)`, `get(id)`,
`addAll(docs)`, `search(query, k, filter)`. `simple-fts` had none of these and
was awkward to inspect in a REPL as a result.

---

## Testing

Both implementations run the same behavioural suite against
`conformance/corpus/`.

**Cross-implementation conformance is the headline test.** `conformance/run.sh`:

1. Kotlin writes an index from the shared corpus with `HashingEmbedder`.
2. Python opens it, runs a fixed query set, emits ranked ids + scores.
3. Python writes its own index from the same corpus.
4. Kotlin opens it and runs the same queries.
5. Assert: `vectors.f32` is byte-identical across both writers; ranked id lists
   match exactly; scores agree within 1e-6.

Commit a golden index in `conformance/fixtures/` generated at format v1, and
test that both implementations still read it. That is the regression guard on
the format itself.

Additional required coverage, drawn from the `simple-fts` review's gaps:

- Null / absent indexed field
- Missing `@SemanticId` → an error naming the annotation and the class, not a
  bare `NoSuchElementException`
- Non-ASCII text (Cyrillic, accented Latin, CJK) round-trips
- Empty and whitespace-only queries
- Re-adding an identical document (should be a no-op via hash)
- Deleting an id that was never added
- `k` larger than `live_count`
- Manifest mismatch on open → refusal, with both ids in the message
- Every stored row is unit-norm

Tests must be isolated. Do not share mutable index state across cases the way
the `simple-fts` specs did.

---

## Build and CI

- Kotlin: Gradle KTS, JDK 22+ (FFM is finalized there), Kotest, `ktlint`.
- Python: `uv`, `pytest`, `ruff`, `mypy --strict`. Dependencies: `numpy` and an
  HTTP client. Nothing else in `core`.
- GitHub Actions, three jobs: `kotlin`, `python`, `conformance`. The
  conformance job needs both toolchains and runs `conformance/run.sh`.
- MIT `LICENSE` in the first commit, added via GitHub's license-template flow
  so the repo gets machine-readable detection. MIT over Apache-2.0 deliberately:
  a teaching repo's license should be short enough that a reader actually reads
  it, and the patent grant buys nothing here. `simple-fts` has no license at
  all, which means the default is all-rights-reserved and the demo technically
  cannot be copied — backport MIT there too.
- Any diagram is a committed SVG, not an external embed. `simple-fts` links its
  data-structure diagram to Google Drive — a quiet link-rot dependency.

---

## Build order

1. `SPEC.md`, MIT `LICENSE`, both build files, CI skeleton.
2. `HashingEmbedder` in both languages. No network dependency in the test
   suite, ever.
3. Format read/write in both. Conformance job green. **This is the milestone
   that proves the project premise — reach it before anything else.**
4. Search: mask, dot, top-k heap. Behavioural suite green.
5. Upsert, tombstones, compaction, incremental hashing.
6. `GeminiEmbedder`.
7. CLI: `index`, `search`, `stats`, `compact`.
8. Benchmarks: dtype curve, filtered-search curve vs an ANN baseline,
   Kotlin-loop vs numpy-BLAS.
9. `LocalEmbedder` and the local-vs-hosted quality delta.

## Later, deliberately out of scope for v1

Reciprocal rank fusion with `simple-fts`, `score = Σ 1/(60 + rank)`. RRF
consumes ranks only, so no score calibration is needed, and it is roughly ten
lines. Both projects being on the JVM makes this a function call rather than a
service boundary. Do not build it until both indexes are independently correct.

---

## Conventions

- Comments explain *why*, particularly for the format's fixed choices
  (endianness, no header, append-only). Anyone reading this code is reading it
  to learn the design.
- Prefer a named constant and a clear error over a clever one-liner.
- No hedging language in the README. Every performance claim is backed by a
  committed benchmark, and benchmark numbers state their conditions. If a
  measurement is noisy, report direction and omit the magnitude.

---

## Provenance

**This document is 100% AI-generated.** No part of it was written or edited by
a human. Nothing in it has been fact-checked, benchmarked, or reviewed by a
person — including the performance claims, the format specification, and the
criticisms of `simple-fts`. Verify before relying on any of it.
