# simple-semantic

Brute-force semantic search. `scores = D @ q`, then top-k. No ANN index, no
vector database.

Two implementations — Kotlin and Python — over one shared on-disk format
([SPEC.md](SPEC.md)). An index written by either is byte-identical to one
written by the other, and each reads the other's output. CI checks it.

A sibling to [`simple-fts`](https://github.com/engineeringthefutureai/simple-fts),
which does the same for inverted-index full-text search.

```
ok    vectors.f32 identical (5120 bytes)
ok    docs.jsonl identical (5786 bytes)
ok    offsets.bin identical (168 bytes)
ok    python reads the kotlin index (12 queries, 110 results)
ok    kotlin reads the python index (12 queries, 110 results)
ok    stories vectors.f32 identical (30720 bytes)
ok    both implementations rank identically on real embeddings
```

---

## The format

An index is a directory. Every file is inspectable with standard tools.

```
index/
  manifest.json      # embedder id, dimension, row counts
  vectors.f32        # row-major little-endian float32, no header
  docs.jsonl         # one JSON object per line; line N describes row N
  offsets.bin        # uint64 LE line offsets, for O(1) document fetch
  tombstones.bits    # one bit per row, LSB-first; absent means no deletions
```

```console
$ jq -c . index/manifest.json
{"format_version":1,"embedder_id":"hashing-0@64","dimension":64,...}

$ od -A d -t f4 -j 32 -N 16 index/vectors.f32
0000032      0.24253562               0      0.24253562               0
```

The invariant everything serves: **line N of `docs.jsonl` describes row N of
`vectors.f32`.** [SPEC.md](SPEC.md) is normative.

`open()` refuses an index whose `embedder_id` differs from the configured
embedder, naming both. Updates are append-only: tombstone a row, append a row.

## Usage

### Python

```python
import asyncio
from simple_semantic import Document, HashingEmbedder, SemanticIndex

async def main():
    index = SemanticIndex.create("./notes.index", HashingEmbedder(dimension=256))

    await index.add_all([
        Document(id="n1", text="Cosine similarity over a dense matrix.",
                 meta={"source": "notes", "tags": ["ir"]}),
        Document(id="n2", text="Sourdough needs a mature starter.",
                 meta={"source": "recipes"}),
    ])

    for hit in await index.search("vector similarity", k=5):
        print(f"{hit.score:+.4f}  {hit.id}")

    await index.upsert(Document(id="n1", text="Revised text."))
    index.delete("n2")
    index.compact()

asyncio.run(main())
```

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
docs = from_lambdas(rows, id_of=lambda r: r["key"], text_of=lambda r: r["body"])
```

### Kotlin

```kotlin
import dev.simplesemantic.*
import java.nio.file.Path
import kotlinx.coroutines.runBlocking

runBlocking {
    SemanticIndex.create(Path.of("./notes.index"), HashingEmbedder(dimension = 256)).use { index ->
        index.addAll(listOf(
            Document("n1", "Cosine similarity over a dense matrix.",
                     mapOf("source" to "notes", "tags" to listOf("ir"))),
            Document("n2", "Sourdough needs a mature starter."),
        ))

        for (hit in index.search("vector similarity", k = 5)) {
            println("%+.4f  %s".format(hit.score, hit.id))
        }

        index.upsert(Document("n1", "Revised text."))
        index.delete("n2")
        index.compact()
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
val documents = documentsFrom(rows, idOf = { it.key }, textOf = { it.body })
```

Both snippets are executed by the test suites (`tests/test_readme.py`,
`ReadmeSpec.kt`), so they cannot drift.

### Command line

```console
$ simple-semantic index ./notes.index -i corpus.jsonl
added 20, replaced 0, skipped 0 (unchanged) -> 20 live rows

$ simple-semantic index ./notes.index -i corpus.jsonl      # nothing changed
added 0, replaced 0, skipped 20 (unchanged) -> 20 live rows

$ simple-semantic search ./notes.index "approximate nearest neighbour" -k 3
  1. +0.560112  ann/hnsw
     Hierarchical navigable small world graphs approximate nearest neighbour search by...
  2. +0.452911  brute/exact
     Exact k nearest neighbour search computes every dot product and keeps the best k,...
  3. +0.288675  emb/norm
     Normalising every row at write time makes cosine similarity a plain dot product...

$ simple-semantic stats ./notes.index
$ simple-semantic compact ./notes.index
```

A chunk whose content hash is unchanged is not re-embedded, so the second
`index` run makes no embedding calls.

## Embedders

| | network | deterministic |
|---|---|---|
| `HashingEmbedder` | no | yes, bit-for-bit across languages |
| `ReplayEmbedder` | no | yes, it is a recording |
| `GeminiEmbedder` | yes | no |

`HashingEmbedder` makes the whole test suite runnable with no API key.

`ReplayEmbedder` serves vectors a real model produced once, recorded to
[a fixture](conformance/fixtures/story-embeddings-v1.json) keyed by
`sha256(text)`:

```python
embedder = ReplayEmbedder.from_file("conformance/fixtures/story-embeddings-v1.json")
index = SemanticIndex.create("./stories.index", embedder)   # gemini-embedding-001@768
```

A miss is an error, never a zero vector. Documents and queries are separate
maps, so asking for a document embedding of a recorded query fails rather than
serving the wrong task type.

Document and query embedding are separate methods and stay separate: Gemini
needs `RETRIEVAL_DOCUMENT` versus `RETRIEVAL_QUERY`, e5/BGE-style models need
`passage: ` / `query: ` prefixes.

## Retrieval quality

Ten stories, twenty queries that describe one without naming it, five negative
controls. Reproduced exactly by `pytest` and `gradlew test`:

```
top-1 accuracy on targeted queries: 18/20      (target in top 3 for 20/20)
targeted top score:  min 0.5538  mean 0.6679  max 0.7466
negative top score:  min 0.5141  mean 0.5342  max 0.5507
whole-matrix range:  0.4508 .. 0.7466          (250 query-document pairs)
```

Exact search cannot abstain — the negative controls still return k results.
Their best score sits below the weakest genuine match, and the suite asserts
that separation.

All 250 similarities fall in a 0.3-wide band far from zero, and where the band
sits moves with the model and the corpus. That is why the API exposes no
absolute score threshold: only relative ordering carries signal.

## Building

Requires **JDK 22 or newer** (the Foreign Function & Memory API is final there)
and **Python 3.11+**.

```console
$ cd kotlin && ./gradlew build               # 73 tests
$ cd python && uv venv .venv && uv pip install -e ".[dev]"
$ cd python && .venv/bin/python -m pytest    # 78 tests
$ ./conformance/run.sh
```

`conformance/run.sh` builds an index with each implementation, compares the
files byte for byte, runs the query set four ways, checks both still read the
committed v1 golden index, and repeats the byte-identity check over the story
corpus with real recorded embeddings.

To drive the story corpus by hand, see
[conformance/stories/README.md](conformance/stories/README.md).

## License

MIT. See [LICENSE](LICENSE).
