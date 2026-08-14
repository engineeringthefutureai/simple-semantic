# simple-semantic on-disk format

**Format version: 1**

Normative. Where an implementation and this document disagree, this document is
correct.

Two implementations (Kotlin, Python) must produce byte-identical files from the
same inputs and must read each other's output. `conformance/run.sh` verifies it.

---

## 1. An index is a directory

```
index/
  manifest.json      # required — the header for everything else
  vectors.f32        # required — the dense matrix
  docs.jsonl         # required — one line per row
  offsets.bin        # required — O(1) line lookup into docs.jsonl
  tombstones.bits    # optional — absent means no deletions
```

Every file is inspectable with standard tools; that is a requirement.

> **Line `N` of `docs.jsonl` describes row `N` of `vectors.f32`, for all `N` in
> `[0, row_count)`.**

Everything else is bookkeeping around that correspondence.

---

## 2. `manifest.json`

UTF-8 JSON, LF-terminated, written with the canonical rules in §7.

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

Keys are written in exactly that order.

| key | type | meaning |
|---|---|---|
| `format_version` | int | `1`. An implementation MUST refuse a version it does not know. |
| `embedder_id` | string | Identity of the model that produced every row. See §2.1. |
| `dimension` | int | Columns in `vectors.f32`. Must be > 0. |
| `normalized` | bool | Always `true` in v1. |
| `chunker_id` | string | Identity of the chunker, part of the content hash. |
| `hash_algorithm` | string | Always `"sha256"` in v1. |
| `row_count` | int | Total rows, live and tombstoned. |
| `live_count` | int | Rows whose tombstone bit is clear. |
| `created_at` | string | RFC 3339 UTC, second precision, `Z` suffix. |
| `updated_at` | string | Same. Bumped on every mutating operation. |

### 2.1 `embedder_id`

**`open()` MUST refuse an index whose manifest `embedder_id` differs from the
configured embedder, and the error MUST name both ids.**

Swapping the model raises nothing on its own: the shapes still match, the dot
products still compute, the results are just meaningless.

The recommended shape is `model-name@dimension`, so a dimension change is also a
mismatch. It is an opaque string to the format; only equality matters.

### 2.2 Checks required on open

1. `format_version` is known.
2. `embedder_id` equals the configured embedder's id.
3. `dimension` equals the configured embedder's dimension.
4. `filesize(vectors.f32) == row_count * dimension * 4`.
5. `filesize(offsets.bin) == (row_count + 1) * 8`.
6. If `tombstones.bits` exists, `filesize == ceil(row_count / 8)`.

Failures are loud. Check 4 catches a truncated write, a partial copy, and a
half-finished compaction.

---

## 3. `vectors.f32`

Flat, row-major, **little-endian IEEE-754 binary32**. No header. Exactly
`row_count * dimension * 4` bytes. Row `i` occupies bytes
`[i * dimension * 4, (i + 1) * dimension * 4)`.

The header is `manifest.json`. Keeping it out of the binary makes the file
directly mappable with no parsing:

- Python: `np.memmap(path, dtype="<f4", mode="r", shape=(row_count, dim))`
- Kotlin: `FileChannel.map(READ_ONLY, 0, size, arena)` → `MemorySegment`

Kotlin uses `MemorySegment` rather than `ByteBuffer`, which caps at 2 GB —
about 175k rows at 3072 dimensions.

**Endianness.** JVM `ByteBuffer` defaults to big-endian, so any JVM path
touching these bytes must set `ByteOrder.LITTLE_ENDIAN` explicitly. The
implementation uses `ValueLayout.JAVA_FLOAT.withOrder(LITTLE_ENDIAN)`; the
platform-default layout is correct on x86 and silently wrong elsewhere.

### 3.1 Normalization

Every row MUST be L2-normalized at write time, whatever the embedder claims.
Because two implementations must produce byte-identical files, the arithmetic is
specified rather than left to the language:

```
ss = 0.0                                  # float64
for i in 0 .. dim-1:                      # strictly sequential, ascending
    ss = ss + (float64)v[i] * (float64)v[i]
norm = sqrt(ss)                           # IEEE-754 correctly-rounded
if norm == 0.0:
    out = e_0                             # 1.0 at index 0, 0.0 elsewhere
else:
    for i in 0 .. dim-1:
        out[i] = (float32)((float64)v[i] / norm)
```

- **Sequential accumulation, ascending index.** Pairwise or blocked summation —
  what a vectorized `sum()` or BLAS `ddot` will do — differs in the last ulp,
  which can change the final float32 rounding. NumPy's `cumsum` is specified as
  sequential, so the Python implementation uses it.
- **float64 intermediate.** float32 accumulation loses enough at 3072 dimensions
  to move results.
- **Zero vector maps to `e_0`.** `0/0 = NaN` would poison every later dot
  product silently.

Implementations MUST assert `abs(norm - 1.0) < 1e-5` for every stored row.

---

## 4. `docs.jsonl`

UTF-8, one JSON object per line, each terminated by a single `LF` including the
last. No `CR`, no blank lines. Line `N` describes row `N`.

```json
{"id":"notes/embeddings.md#3","text":"…","meta":{"source":"notes/embeddings.md","tags":["ml"]},"hash":"9f86d0…"}
```

Keys in exactly this order: `id`, `text`, `meta`, `hash`. All four required;
`meta` may be `{}`.

| key | type | meaning |
|---|---|---|
| `id` | string | Logical document id. Non-empty. Not unique across rows — see §6. |
| `text` | string | The chunk text that was embedded. |
| `meta` | object | Caller metadata, returned with results. Value types restricted, see §7.2. |
| `hash` | string | Lowercase hex `sha256`, see §4.1. |

### 4.1 The content hash

```
hash = sha256( utf8(text) || 0x00 || utf8(embedder_id) || 0x00 || utf8(chunker_id) )
```

Hex-encoded lowercase. The NUL separators are unambiguous because none of the
three inputs may contain a NUL byte.

On re-index, a chunk whose hash matches the one already stored under that id is
not re-embedded. The embedder and chunker ids are inside the hash because
changing either invalidates the vector even when the text is identical.

---

## 5. `offsets.bin`

`row_count + 1` **unsigned 64-bit little-endian** integers, no header. Exactly
`(row_count + 1) * 8` bytes.

Entry `i` is the byte offset where line `i` begins; entry `row_count` is the
length of `docs.jsonl`. Line `i` is the bytes `[offsets[i], offsets[i+1])`,
trailing `LF` included and stripped by the reader. Entries strictly increase.

Fetching a result's document is a seek and a read of known length rather than a
scan.

---

## 6. `tombstones.bits`

`ceil(row_count / 8)` bytes, no header. **Bit `i` is bit `(i mod 8)` of byte
`floor(i / 8)`, counting from the least significant bit.** Set means row `i` is
deleted.

An absent file means no deletions. Bits beyond `row_count` in the final byte are
padding, MUST be written as `0`, and are ignored on read.

LSB-first is stated explicitly because MSB-first produces a file of the same
size that parses without error and disagrees about which rows are deleted.

### Update semantics: append-only

**A row is never mutated in place.**

`upsert(doc)`:
1. If `id` maps to a live row, set that row's tombstone bit.
2. Append the new vector as a new row of `vectors.f32`.
3. Append the new document to `docs.jsonl`, extend `offsets.bin`.

`delete(id)` sets the tombstone bit of the live row for `id`. Deleting an
unknown id is a no-op, not an error.

The `id → row` map is rebuilt by scanning `docs.jsonl` on load: for each row in
ascending order, if the row is live, `map[id] = row`. Because appends are
ascending and the previous row was tombstoned first, the latest live row wins.

### `compact()`

Rewrites all files dropping tombstoned rows and renumbering; live rows keep
their relative order. It MUST be crash-safe:

1. Write the complete new index into a sibling temporary directory.
2. `fsync` each file, then the directory.
3. Atomically rename the old aside, rename the new into place, delete the old.

---

## 7. Canonical JSON

Both `manifest.json` and every line of `docs.jsonl` are written with these
rules, so that two implementations emit the same bytes.

1. No insignificant whitespace. Separators are exactly `,` and `:`.
2. Object keys are emitted in the order specified for that object (§2, §4). For
   `meta`, whose keys are user-supplied, keys sort **ascending by Unicode code
   point**.
3. Strings are UTF-8. `\"`, `\\`, `\b`, `\f`, `\n`, `\r`, `\t` where applicable;
   any other character below `U+0020` as `\u00XX` with **lowercase** hex. Every
   other character, including all non-ASCII, is emitted literally. `/` is never
   escaped.
4. `true`, `false`, `null` are lowercase bare literals.

Python's `json.dumps(obj, ensure_ascii=False, separators=(",", ":"))` satisfies
these for the permitted types. Kotlin hand-writes an encoder, because no JVM
JSON library guarantees all four by default.

### 7.2 Permitted `meta` value types

- string, boolean, null
- integer in the signed 64-bit range
- array of permitted values
- object with string keys and permitted values

**Floating-point numbers are not permitted in v1.** Their shortest round-trip
decimal representation is not agreed on across languages, which would break
byte-identity. Store a float as a string.

Implementations MUST reject a disallowed value at write time, naming the
offending key.

---

## 8. Search

Not a file-format concern, but the ordering guarantees are part of what
conformance checks.

```
scores = vectors[live_rows] · query
top_k  = bounded_min_heap(scores, k)
```

- The query vector is L2-normalized by §3.1, so a score is a cosine similarity.
- Top-k uses a bounded heap, never a full sort of `N`.
- **Ties break by ascending row index.**
- `k` larger than the live count returns all live rows.
- Scores accumulate in float64 from float32 inputs. Unlike §3.1 the reduction
  order is **not** pinned: BLAS, a vectorized reduction or a sequential loop are
  all permitted. Stored bytes must be identical; scores need only agree to
  `1e-6`.
- Because the reduction order is free, the tie-break is what keeps ranked lists
  identical. A selection that keeps an arbitrary subset of the rows tied on the
  k-th score — which a bare `argpartition` or `nth_element` does — returns a
  different *set*, not merely a different order. Take every row strictly better
  than the k-th score, then fill from the tied rows in ascending row order.

The API MUST NOT expose an absolute similarity threshold. Cosine similarities
cluster in a narrow, model- and corpus-dependent band; only relative ordering
carries signal.

---

## 9. Versioning

`format_version` increments when a change would make an older reader wrong — a
new file, a changed field meaning, a different normalization rule. Adding an
optional key an older reader can ignore does not require a bump, but does
require a note here.

`conformance/fixtures/` holds a golden index generated at v1. Both
implementations must keep reading it. That fixture, not the code, is the
regression guard on this document.

---

## Appendix A: the hashing embedder

Not part of the format, but conformance depends on both implementations
producing identical vectors for identical text.

`embedder_id` is `hashing-<seed>@<dimension>`.

```
tokens = tokenize(text)
acc    = float64[dimension], all zero
seed8  = seed as 8 unsigned little-endian bytes

for token in tokens:                      # source order, duplicates included
    h      = sha256( utf8(token) || 0x00 || seed8 )
    column = uint32_le(h[0:4]) mod dimension
    sign   = +1.0 if (h[4] AND 1) == 0 else -1.0
    acc[column] = acc[column] + sign

row = float32[dimension] from acc         # exact: every value is a small integer
out = normalize(row)                      # §3.1
```

### Tokenization

```
lowercase(text) with locale-independent full Unicode case mapping
split into maximal runs of characters in Unicode general categories L* or N*
```

- **`\p{L}\p{N}`, not `[a-z0-9]`.** An ASCII-only class returns an empty token
  list for every non-Latin script.
- **Locale-independent casing.** Java requires `Locale.ROOT` explicitly; Python's
  `str.lower()` already is. Under a Turkish locale `I` maps to `ı`, so the same
  corpus would index differently per machine. Both languages implement the same
  SpecialCasing rules, including Greek final sigma (`ΟΔΟΣ` → `οδος` with U+03C2)
  and `İ` → `i` + U+0307.
- **Combining marks are not token characters.** Category Mn is outside
  `\p{L}\p{N}`, so a combining mark ends a token in both implementations.

Python's `[^\W_]+` with `re.UNICODE` is equivalent to `[\p{L}\p{N}]+`.

Documents and queries share the embedding function here, so search over a hashed
index still returns sensible neighbours.

### Command-line encoding

The JVM decodes `main`'s arguments using `sun.jnu.encoding`, derived from the
process locale and **not** affected by `-Dfile.encoding`. Under the default POSIX
locale that is US-ASCII, so a Cyrillic query reaches the tokenizer as question
marks and returns an arbitrary but well-formed ranking. Run the CLI under a
UTF-8 locale; the Kotlin CLI warns when it detects otherwise.

---

## Appendix B: the replay fixture

`conformance/fixtures/story-embeddings-v1.json` records a real model's output
once, so retrieval tests run offline with identical numbers everywhere.

```json
{
  "fixture_version": 1,
  "embedder_id": "gemini-embedding-001@768",
  "dimension": 768,
  "normalized": false,
  "documents": [{"id": "story-01", "key": "<sha256>", "vector": [...]}],
  "queries":   [{"key": "<sha256>", "vector": [...]}]
}
```

- **`embedder_id` is required and checked against `dimension`.** An index built
  from an anonymous fixture inherits an embedder id that means nothing.
- **`key` is `sha256(utf8(text))` of the exact text embedded**, not a document
  id. An edited document stops matching its stale vector rather than silently
  keeping it.
- **Documents and queries are separate maps.** They were embedded with different
  task types, so the same string has two different correct vectors.
- **Vectors are stored unnormalized, as returned.** `gemini-embedding-001`
  pre-normalizes only its default 3072-dimension output; at 768 the norms land
  near 0.59. Keeping them that way is what exercises §3.1 against real input.

A lookup miss MUST be an error naming the mode, the text and the fixture. It
MUST NOT return a zero vector or anything else plausible.

Because the vectors are arbitrary decimals rather than small integers, replaying
them through both implementations is a stronger byte-identity check than the
hashing embedder provides: decimal parsing, float32 narrowing and summation
order all have to agree.
