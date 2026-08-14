# simple-semantic

Brute-force semantic search. No ANN index, no vector database, no hidden
machinery. A sibling to
[`simple-fts`](https://github.com/engineeringthefutureai/simple-fts), which
does the same thing for inverted-index full-text search.

Two implementations — Kotlin and Python — over **one shared on-disk format**.
The format ([SPEC.md](SPEC.md)) is the primary artifact. The implementations
exist to prove it is real: an index written by either is byte-identical to one
written by the other, and each reads the other's output. CI checks this on
every push.

```
==> comparing files byte for byte
ok    vectors.f32 identical (5120 bytes)
ok    docs.jsonl identical (5786 bytes)
ok    offsets.bin identical (168 bytes)
ok    manifest.json agrees on every non-timestamp field
==> running the query set four ways
ok    python reads the kotlin index (12 queries, 110 results)
ok    kotlin reads the python index (12 queries, 110 results)
ok    both implementations rank identically (12 queries, 110 results)
```

---

## Thesis

Exact k-NN over a dense matrix is `scores = D @ q` followed by a top-k
selection. That is the whole algorithm. Everything a vector database adds is
either an approximation of this or a preprocessing step around it.

At the scale most people actually operate — 10k to 1M chunks — brute force is
not merely "good enough". It is better on three axes:

**1. Filtered search is exact.** A metadata predicate is a boolean mask
applied before the dot products, so the filter *reduces* work. An ANN index
must choose between pre-filtering, which breaks graph connectivity and strands
traversal in dead ends, and post-filtering, which fetches the top 1000 and
hopes enough survive. Neither degrades gracefully as selectivity rises.

**2. Updates and deletes are trivial.** Tombstone a row, append a row. No
graph repair, no index rebuild. See [SPEC.md §6](SPEC.md).

**3. Recall is 100% by construction.** ANN trades recall for latency. That is
a real cost, and it is absent from most marketing. Here it needs no tuning
parameter and no measurement: every live row is scored, so the returned top-1
*is* the top-1.

There is also a fourth thing, which is not about performance:

**4. Opening an index with the wrong embedder is an error, not a surprise.**
The manifest records the model identity, and `open()` refuses a mismatch by
name. A model swap against an existing index otherwise raises nothing and
crashes nothing — the vectors still have the right shape, the dot products
still compute, the top-k still returns k results. They are simply meaningless.
Most vector databases do not guard this.

### The honest limits

- **The hosted embedding model is the one thing here that is not neanderthal.**
  It is a network call to someone else's GPU. It is isolated behind
  [one interface](kotlin/core/src/main/kotlin/dev/simplesemantic/Embedder.kt)
  so the boundary is visible, and a local ONNX implementation is planned so the
  gap can be measured rather than asserted.
- **The Python search path delegates to BLAS.** `D @ q` through NumPy is the
  least neanderthal line in the project. The Kotlin side is a hand-written
  `FloatArray` dot loop, which is the point *there*.
- **The performance claims above are structural, not yet benchmarked.** The
  filtered-search curve against an ANN baseline, the fp32/fp16/int8 dtype
  curve, and Kotlin-loop-versus-BLAS are all still to be measured. Nothing in
  this README states a number that is not printed by something in this repo.

---

## The format in one screen

An index is a **directory**, not a file. Everything in it is inspectable with
standard tools — that is a design requirement, not a nicety.

```
index/
  manifest.json      # embedder id, dimension, row counts. The header.
  vectors.f32        # row-major little-endian float32. No header at all.
  docs.jsonl         # one JSON object per line; line N describes row N
  offsets.bin        # uint64 LE line offsets, for O(1) document fetch
  tombstones.bits    # one bit per row, LSB-first. Absent means no deletions.
```

```console
$ jq -c . index/manifest.json
{"format_version":1,"embedder_id":"hashing-0@64","dimension":64,...}

$ head -c 200 index/docs.jsonl
{"id":"ann/hnsw","text":"Hierarchical navigable small world graphs...

$ od -A d -t x1 -j 32 -N 16 index/vectors.f32
0000032 42 5b 78 3e 00 00 00 00 42 5b 78 3e 00 00 00 00

$ od -A d -t f4 -j 32 -N 16 index/vectors.f32     # 0x3e785b42 little-endian
0000032      0.24253562               0      0.24253562               0
```

The one invariant everything else serves: **line N of `docs.jsonl` describes
row N of `vectors.f32`.** [SPEC.md](SPEC.md) is normative and explains every
choice that looks arbitrary — why there is no binary header, why the tombstone
bit order is stated explicitly, why floats are banned from metadata, and why
the normalization arithmetic is specified down to the summation order.

---

## Usage

### Python

```python
import asyncio
from simple_semantic import Document, HashingEmbedder, SemanticIndex

async def main():
    embedder = HashingEmbedder(dimension=256)
    index = SemanticIndex.create("./notes.index", embedder)

    await index.add_all([
        Document(id="n1", text="Cosine similarity over a dense matrix.",
                 meta={"source": "notes", "tags": ["ir"]}),
        Document(id="n2", text="Sourdough needs a mature starter.",
                 meta={"source": "recipes"}),
    ])

    for hit in await index.search("vector similarity", k=5):
        print(f"{hit.score:+.4f}  {hit.id}")

    # Exact metadata filtering, applied before the dot products.
    await index.search("vector similarity", k=5,
                       filter=lambda meta: meta.get("source") == "notes")

    await index.upsert(Document(id="n1", text="Revised text."))  # tombstone + append
    index.delete("n2")
    index.compact()                                              # crash-safe rewrite

asyncio.run(main())
```

Both snippets above are executed verbatim by the test suites
(`tests/test_readme.py`, `ReadmeSpec.kt`), so they cannot drift from the
real API.

Extraction from your own types, both entry points public:

```python
from dataclasses import dataclass
from typing import Annotated
from simple_semantic import SemanticId, SemanticIndexed, SemanticMeta, from_dataclass, from_lambdas

@dataclass
class Note:
    id: Annotated[str, SemanticId]
    title: Annotated[str, SemanticIndexed]
    body: Annotated[str | None, SemanticIndexed]   # nullable is fine
    tags: Annotated[list[str], SemanticMeta]

docs = from_dataclass(notes)
# or, with no annotations and no requirement to be a dataclass:
docs = from_lambdas(rows, id_of=lambda r: r["key"], text_of=lambda r: r["body"])
```

### Kotlin

```kotlin
import dev.simplesemantic.*
import java.nio.file.Path
import kotlinx.coroutines.runBlocking

// addAll, upsert and search are suspend: the embedder behind them may be a
// network call, and hiding that behind a synchronous signature would be a lie.
runBlocking {
    val embedder = HashingEmbedder(dimension = 256)
    SemanticIndex.create(Path.of("./notes.index"), embedder).use { index ->
        index.addAll(listOf(
            Document("n1", "Cosine similarity over a dense matrix.",
                     mapOf("source" to "notes", "tags" to listOf("ir"))),
            Document("n2", "Sourdough needs a mature starter."),
        ))

        for (hit in index.search("vector similarity", k = 5)) {
            println("%+.4f  %s".format(hit.score, hit.id))
        }

        index.search("vector similarity", k = 5) { it["source"] == "notes" }

        index.upsert(Document("n1", "Revised text."))   // tombstone + append
        index.delete("n2")
        index.compact()                                 // crash-safe rewrite
    }
}
```

```kotlin
data class Note(
    @SemanticId val id: String,
    @SemanticIndexed(order = 0) val title: String,
    @SemanticIndexed(order = 1) val body: String?,   // nullable is fine
    @SemanticMeta val tags: List<String>,
)

val documents = documentsFrom(notes)
// or, with plain lambdas — public, not internal:
val documents = documentsFrom(rows, idOf = { it.key }, textOf = { it.body })
```

### Command line

Both implementations ship the same four commands.

```console
$ simple-semantic index ./notes.index -i corpus.jsonl
added 20, replaced 0, skipped 0 (unchanged) -> 20 live rows

$ simple-semantic index ./notes.index -i corpus.jsonl      # re-index, nothing changed
added 0, replaced 0, skipped 20 (unchanged) -> 20 live rows

$ simple-semantic search ./notes.index "approximate nearest neighbour" -k 3
  1. +0.560112  ann/hnsw
     Hierarchical navigable small world graphs approximate nearest neighbour search by...
  2. +0.452911  brute/exact
     Exact k nearest neighbour search computes every dot product and keeps the best k,...
  3. +0.288675  emb/norm
     Normalising every row at write time makes cosine similarity a plain dot product...

$ simple-semantic stats ./notes.index
path: ./notes.index
embedder_id: hashing-0@64
dimension: 64
row_count: 20
live_count: 20
deleted_count: 0
vectors_bytes: 5120

$ simple-semantic compact ./notes.index
dropped 0 tombstoned rows (20 -> 20)
```

The second `index` run is the feature that decides whether this is usable
daily: a chunk whose content hash is unchanged is not re-embedded, so
re-indexing after editing one file costs one embedding call rather than
twenty thousand.

---

## Embedders

| | network | deterministic | use |
|---|---|---|---|
| `HashingEmbedder` | no | yes, bit-for-bit across languages | tests, conformance, CI |
| `GeminiEmbedder` | yes | no | real retrieval |
| `LocalEmbedder` | no | — | not yet built |

`HashingEmbedder` is required, not optional: it is what makes the entire test
suite and the conformance job runnable with no API key. A test suite that
needs a credential is a test suite that stops being run.

Document and query embedding are **separate methods and stay separate**.
Gemini needs `RETRIEVAL_DOCUMENT` versus `RETRIEVAL_QUERY`; e5/BGE-style models
need `passage: ` / `query: ` prefixes. A single `embed()` makes that asymmetry
unrepresentable and losing it costs retrieval quality without raising anything.

Two Gemini quirks are documented in the code because they are silent when you
get them wrong: `gemini-embedding-001` only pre-normalizes its default
3072-dimension output, and `gemini-embedding-002` collapses a multi-input
request into a single embedding unless each input is wrapped individually.

---

## No score thresholds

The API does not expose an absolute similarity threshold and will not grow one.
Embedding spaces are anisotropic: cosine similarities cluster in a narrow band
that shifts with model and corpus, so `score > 0.7` means something different
for every model and nothing at all in general. Only relative ordering carries
signal.

---

## Building

Requires **JDK 22 or newer** (the Foreign Function & Memory API is final there;
see [SPEC.md §3](SPEC.md)) and **Python 3.11+**.

```console
$ cd kotlin && ./gradlew build          # 62 tests
$ cd python && uv venv .venv && uv pip install -e ".[dev]"
$ cd python && .venv/bin/python -m pytest    # 64 tests
$ ./conformance/run.sh                  # the one that matters
```

`conformance/run.sh` builds an index with each implementation from the shared
corpus, compares the files byte for byte, runs the query set four ways
(each implementation against each index), and finally checks that both still
read the committed v1 golden index in `conformance/fixtures/`. That fixture,
not the code, is the regression guard on the format.

---

## Not in this repository, on purpose

ANN indexing of any kind. A dependency on Chroma, FAISS, Qdrant, Milvus,
pgvector or LanceDB. A vendored BLAS on the Kotlin side. Sharding, or any
network protocol between the two implementations. LLM generation, agents, or a
RAG chain — this is retrieval only.

Deliberately deferred: reciprocal rank fusion with `simple-fts`
(`score = Σ 1/(60 + rank)`, roughly ten lines, needs no score calibration
because it consumes ranks only). Not until both indexes are independently
correct.

## License

MIT. See [LICENSE](LICENSE).
