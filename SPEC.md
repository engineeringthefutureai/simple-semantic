# simple-semantic on-disk format

**Format version: 1**

This document is normative. Where an implementation and this document
disagree, this document is correct and the implementation has a bug.

The format is the primary artifact of this project. Two implementations
(Kotlin, Python) exist to prove the format is real: an index written by
either must be byte-identical to one written by the other from the same
inputs, and each must read the other's output. `conformance/run.sh` verifies
this in CI.

Every design choice below that looks arbitrary is explained. Anyone reading
this is reading it to learn the design.

---

## 1. An index is a directory

Not a single file. Four files, one optional:

```
index/
  manifest.json      # required — the header for everything else
  vectors.f32        # required — the dense matrix
  docs.jsonl         # required — one line per row
  offsets.bin        # required — O(1) line lookup into docs.jsonl
  tombstones.bits    # optional — absent means no deletions
```

Each file is inspectable with standard tools. That is a design requirement,
not a nicety: `head docs.jsonl`, `jq . manifest.json`, and
`xxd vectors.f32 | head` must all do something useful without a library.

### The core invariant

> **Line `N` of `docs.jsonl` describes row `N` of `vectors.f32`, for all
> `N` in `[0, row_count)`.**

Everything else in this format is bookkeeping around that one correspondence.
An implementation that can break it is wrong.

---

## 2. `manifest.json`

UTF-8 JSON object, LF-terminated. Written with the canonical JSON rules in
§7 so that two implementations produce identical bytes for identical
content (excluding timestamps).

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

Keys are written in exactly the order above.

| key | type | meaning |
|---|---|---|
| `format_version` | int | `1`. An implementation MUST refuse a version it does not know. |
| `embedder_id` | string | Identity of the model that produced every row. See §2.1. |
| `dimension` | int | Columns in `vectors.f32`. Must be > 0. |
| `normalized` | bool | Always `true` in v1. Rows are L2-normalized at write time. |
| `chunker_id` | string | Identity of the chunker, part of the content hash. |
| `hash_algorithm` | string | Always `"sha256"` in v1. |
| `row_count` | int | Total rows, live and tombstoned. |
| `live_count` | int | Rows whose tombstone bit is clear. Derivable; stored for O(1) `stats`. |
| `created_at` | string | RFC 3339 UTC, second precision, `Z` suffix. |
| `updated_at` | string | Same. Bumped on every mutating operation. |

### 2.1 `embedder_id` is load-bearing

**`open()` MUST refuse to serve an index whose manifest `embedder_id`
differs from the configured embedder, and the error MUST name both ids.**

This is the single most important correctness rule in the project.

Swapping the embedding model against an existing index raises no exception
and produces no crash anywhere in the pipeline. The vectors still have the
right shape, the dot products still compute, the top-k heap still returns
`k` results. The results are simply meaningless — two unrelated coordinate
systems compared as though they were one. It is a silent, total failure that
looks exactly like a working system.

Most vector databases do not guard this. This one does.

The recommended `embedder_id` shape is `model-name@dimension`, so a
dimension change is also a mismatch. It is an opaque string to the format;
only equality matters.

### 2.2 Checks required on open

1. `format_version` is known.
2. `embedder_id` equals the configured embedder's id (§2.1).
3. `dimension` equals the configured embedder's dimension.
4. `filesize(vectors.f32) == row_count * dimension * 4`. Fail loudly on
   mismatch — this catches a truncated write, a partial copy, and a
   half-finished compaction.
5. `filesize(offsets.bin) == (row_count + 1) * 8`.
6. If `tombstones.bits` exists, `filesize == ceil(row_count / 8)`.

---

## 3. `vectors.f32`

Flat, row-major, **little-endian IEEE-754 binary32**. No header. Exactly
`row_count * dimension * 4` bytes. Row `i` occupies bytes
`[i * dimension * 4, (i + 1) * dimension * 4)`.

### Why no header

The header is `manifest.json`. Keeping it out of the binary makes the file
directly mappable with zero parsing on both platforms:

- Python: `np.memmap(path, dtype="<f4", mode="r", shape=(row_count, dim))`
- Kotlin: `FileChannel.map(READ_ONLY, 0, size, arena)` → `MemorySegment`

A 16-byte header would mean an offset argument threaded through every read
path and an off-by-one waiting to happen in exactly one of the two
implementations.

### Kotlin: use FFM, not ByteBuffer

`MemorySegment` and `Arena`, not `ByteBuffer`. `ByteBuffer` caps at 2 GB,
which at 3072 dimensions is roughly 175k rows — a limit a real knowledge
base reaches, at which point the fix is a redesign rather than a patch. FFM
maps past it.

### Endianness

The format is little-endian. JVM `ByteBuffer` defaults to **big**-endian, so
any JVM code path touching these bytes must set `ByteOrder.LITTLE_ENDIAN`
explicitly. `MemorySegment` requires an explicit `ValueLayout`, which is why
the implementation uses `ValueLayout.JAVA_FLOAT.withOrder(LITTLE_ENDIAN)`
rather than the platform-default layout — the latter is correct on x86 and
silently wrong on a big-endian JVM.

There is a conformance test that fails if this regresses.

### 3.1 Normalization is mandatory and exactly specified

Every row MUST be L2-normalized at write time, regardless of what the
embedder claims about its output. Normalization is idempotent, costs one
pass, and makes the index immune to a model that quietly changes its
output convention.

Because two implementations must produce **byte-identical** files, the
arithmetic is specified, not left to the language:

```
# v is the input row, dim floats
ss = 0.0                                  # float64
for i in 0 .. dim-1:                      # strictly sequential, ascending
    ss = ss + (float64)v[i] * (float64)v[i]
norm = sqrt(ss)                           # IEEE-754 correctly-rounded sqrt
if norm == 0.0:
    out = e_0                             # 1.0 at index 0, 0.0 elsewhere
else:
    for i in 0 .. dim-1:
        out[i] = (float32)((float64)v[i] / norm)
```

Three details matter:

- **Sequential accumulation, ascending index.** Pairwise and blocked
  summation — what a vectorized `sum()` or a BLAS `ddot` will do — give a
  different last ulp, which can change the final `float32` rounding. NumPy's
  `cumsum` is specified as sequential, which is why the Python
  implementation uses it rather than `np.sum` or `np.dot`.
- **float64 intermediate.** Accumulating in float32 loses enough precision
  at 3072 dimensions to move results.
- **Zero vector maps to `e_0`.** A zero row has no direction, and a
  `0/0 = NaN` row poisons every subsequent dot product silently. `e_0` is an
  arbitrary but defined choice: it makes the row a valid unit vector that
  will simply rank poorly.

Implementations MUST assert `abs(norm - 1.0) < 1e-5` for every stored row in
their test suites.

---

## 4. `docs.jsonl`

UTF-8, one JSON object per line, each line terminated by a single `LF`
(`0x0A`), including the last. No `CR`. No blank lines. Line `N` describes
row `N` (§1).

```json
{"id":"notes/embeddings.md#3","text":"…","meta":{"source":"notes/embeddings.md","tags":["ml"]},"hash":"9f86d0…"}
```

Keys are written in exactly this order: `id`, `text`, `meta`, `hash`. All
four are required; `meta` may be `{}`.

| key | type | meaning |
|---|---|---|
| `id` | string | Logical document id. Non-empty. Not unique across rows — see §5. |
| `text` | string | The chunk text that was embedded. |
| `meta` | object | Filter/display metadata. Value types restricted, see §7.2. |
| `hash` | string | Lowercase hex `sha256`, see §4.1. |

### 4.1 The content hash

```
hash = sha256( utf8(text) || 0x00 || utf8(embedder_id) || 0x00 || utf8(chunker_id) )
```

Hex-encoded lowercase. The `0x00` separators are unambiguous because none of
the three inputs may contain a NUL byte.

The hash drives incremental indexing: on re-index, a chunk whose hash matches
the hash already stored under that id is **not re-embedded**. This is the
feature that decides whether the tool is usable daily rather than
occasionally, and it is the API cost control — re-indexing a 12k-chunk
knowledge base after editing one file should cost one embedding call, not
12,043.

The embedder and chunker ids are inside the hash because changing either
invalidates the vector even when the text is identical.

---

## 5. `offsets.bin`

`row_count + 1` **unsigned 64-bit little-endian** integers, no header.
Exactly `(row_count + 1) * 8` bytes.

Entry `i` is the byte offset in `docs.jsonl` where line `i` begins. Entry
`row_count` is the total length of `docs.jsonl`. Therefore line `i` is the
bytes `[offsets[i], offsets[i+1])`, with the trailing `LF` included and
stripped by the reader.

This exists so that fetching the document for a search result is a seek and
a read of known length, rather than a scan. Without it, returning 10 results
from a 12k-line file means parsing 12k lines of JSON.

Entries are strictly increasing (every line contains at least `{}` plus the
`LF`).

---

## 6. `tombstones.bits`

`ceil(row_count / 8)` bytes, no header. **Bit `i` is bit `(i mod 8)` of byte
`floor(i / 8)`, counting from the least significant bit.** Set means row `i`
is deleted.

An absent file means no deletions — a fresh index need not write one. Bits
beyond `row_count` in the final byte are padding and MUST be written as `0`
and ignored on read.

LSB-first is stated explicitly because MSB-first is an equally common
convention and the two produce files that are the same size, parse without
error, and disagree about which rows are deleted.

### Update semantics: append-only

**A row is never mutated in place.**

`upsert(doc)`:
1. If `id` currently maps to a live row, set that row's tombstone bit.
2. Append the new vector as a new row of `vectors.f32`.
3. Append the new document as a new line of `docs.jsonl`, extend
   `offsets.bin`.
4. The in-memory `id → row` map now points at the new row.

`delete(id)`: set the tombstone bit of the live row for `id`, if any.
Deleting an unknown id is a no-op, not an error.

The `id → row` map is rebuilt by scanning `docs.jsonl` on load: for each row
in ascending order, if the row is live, `map[id] = row`. Because appends are
ascending and the previous row was tombstoned in step 1, the latest live row
wins.

This structure makes a whole class of bug impossible rather than merely
tested-against. The matrix holds no documents, only rows; the row index *is*
the identity. There is no second copy of a document that an update could
miss. (The sibling project `simple-fts` stored whole document objects inside
its inverted index, and an update refreshed only the postings reachable via
newly added terms — leaving stale copies alive under shared terms. That
shape of defect has nowhere to live here.)

### `compact()`

Rewrites all files, dropping tombstoned rows and renumbering. Live rows keep
their relative order.

It MUST be crash-safe:

1. Write the complete new index into a sibling temporary directory.
2. `fsync` each file, then the directory.
3. Atomically rename the old directory aside, rename the new one into place,
   then delete the old.

A compaction that truncates the original in place and then fails leaves an
index whose `vectors.f32` length disagrees with `row_count` — which the
check in §2.2 catches, but only after the data is gone.

---

## 7. Canonical JSON

Both `manifest.json` and every line of `docs.jsonl` are written with these
rules. They exist so that two implementations in two languages emit the same
bytes.

1. No insignificant whitespace. Separators are exactly `,` and `:`.
2. Object keys are emitted in the order specified for that object
   (§2, §4). For `meta`, whose keys are user-supplied, keys are sorted
   **ascending by Unicode code point**.
3. Strings are UTF-8. The escapes `\"`, `\\`, `\b`, `\f`, `\n`, `\r`, `\t`
   are used where applicable; any other character below `U+0020` is escaped
   as `\u00XX` with **lowercase** hex digits. Every other character,
   including all non-ASCII, is emitted literally. `/` is never escaped.
4. `true`, `false`, `null` are lowercase bare literals.

Python's `json.dumps(obj, ensure_ascii=False, separators=(",", ":"))`
satisfies these rules for the permitted value types. The Kotlin
implementation hand-writes an encoder to match, because no JVM JSON library
guarantees all four points by default.

### 7.2 Permitted `meta` value types

`meta` is a JSON object whose values MUST be one of:

- string
- boolean
- null
- integer in the signed 64-bit range
- array of permitted values
- object with string keys and permitted values

**Floating-point numbers are not permitted in `meta` in format v1.** Not
because they are hard to store, but because their *shortest round-trip
decimal representation* is not agreed on across languages — Python, Kotlin,
and JavaScript will not always print the same digits for the same double,
which would break byte-identity for no benefit. Store a float as a string if
you need one; the format does not filter on numeric ranges anyway.

Implementations MUST reject a disallowed `meta` value at write time with an
error naming the offending key, rather than writing a file the sibling
implementation cannot reproduce.

---

## 8. Search

Not a file-format concern, but the format exists to serve it, and the
ordering guarantees below are part of what conformance checks.

```
mask   = live_rows AND metadata_predicate
scores = vectors[mask] · query
top_k  = bounded_min_heap(scores, k)
```

- The query vector is L2-normalized by the same procedure as §3.1, so a
  score is a cosine similarity in `[-1, 1]`.
- Metadata filtering is a boolean mask applied **before** the dot products.
  The filter therefore *reduces* work: exact, and cheaper as it gets more
  selective.
- Top-k uses a bounded heap, never a full sort of `N`.
- **Ties break by ascending row index.** Two rows with bit-identical scores
  must come back in the same order from both implementations, or conformance
  is comparing noise.
- `k` larger than the live count returns all live rows.
- Scores are compared as float64 computed from float32 inputs, accumulated
  sequentially — same reasoning as §3.1. Conformance allows `1e-6` absolute
  divergence on scores but requires **exact** agreement on the ranked id
  list.

### No absolute score thresholds

The API MUST NOT expose or hardcode an absolute similarity threshold.
Embedding spaces are anisotropic: cosine similarities cluster in a narrow,
model- and corpus-dependent band, so `score > 0.7` means something different
for every model and nothing at all in general. Only relative ordering
carries signal.

---

## 9. Versioning this document

`format_version` increments when a change would make an older reader wrong
— a new file, a changed field meaning, a different normalization rule.
Adding an optional key that an older reader can ignore does not require a
bump, but does require a note here.

`conformance/fixtures/` holds a golden index generated at v1. Both
implementations must keep reading it. That fixture, not the code, is the
regression guard on this document.
