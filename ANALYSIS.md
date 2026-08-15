# simple-semantic: a deep read

An independent analysis of the codebase as of `7138942`. Everything below was
verified by building and running both implementations, not by reading alone.

---

## Verification status

| gate | result |
|---|---|
| `python -m pytest` | **78 passed** |
| `ruff check` / `ruff format --check` | clean (incl. `conformance/`) |
| `mypy` (strict) | clean, 12 source files |
| `./gradlew build` | **73 tests, BUILD SUCCESSFUL** |
| `./conformance/run.sh` | **conformance passed**, all 17 checks |

Every claim the README makes is true. Byte-identity holds on the hashing corpus
*and* on the real recorded Gemini embeddings; both implementations cross-read
each other's output; the committed v1 golden index still ranks as it did.

One environment note: the box had JDK 21, below the JDK 22 floor. The build fails
with a deliberate, accurate message from `build.gradle.kts` rather than a
confusing `cannot find symbol java.lang.foreign.Arena`. I installed JDK 25 to
verify the Kotlin half; it builds and passes there.

---

## What this codebase gets right

This is a genuinely disciplined project, and the discipline is in the right
place. Three things stand out.

**The spec is normative and the code is subordinate to it.** `SPEC.md` doesn't
describe what the code does — it says what the code must do, and the code carries
`SPEC.md §N` references back to the clause it implements. The spec argues for its
own choices (why NUL separators, why LSB-first, why `e_0` for the zero vector),
so a future reader can tell an arbitrary decision from a load-bearing one. That
distinction is what usually rots first.

**The hard parts are the specified parts.** §3.1 pins normalization arithmetic
down to accumulation order and intermediate precision, because that is exactly
where two languages drift in the last ulp. §8 then *deliberately* leaves the
scoring reduction order free — and pins the tie-break instead, because the
tie-break is what actually keeps ranked lists identical once the reduction is
unconstrained. The note that a bare `argpartition`/`nth_element` returns a
different *set* rather than a different *order* is a subtle bug most projects
ship. Both implementations handle it, differently and correctly:

```python
threshold  = scores[np.argpartition(-scores, limit - 1)[limit - 1]]
better     = np.flatnonzero(scores > threshold)
tied       = np.flatnonzero(scores == threshold)
candidates = np.concatenate([better, tied[: limit - better.size]])
```

I confirmed the tie behaviour on five identical vectors: rows `[0, 1, 2]` at
score `1.0`, as specified.

**The failure modes chosen are loud ones.** `open()` refuses an `embedder_id`
mismatch naming both ids; `ReplayEmbedder` makes a miss an error naming the mode,
the text and the fixture rather than returning a plausible zero vector; float
values in `meta` are rejected at *write* time, because by read time the bad file
already exists. The `README`/`SPEC` refusal to expose an absolute similarity
threshold — justified with measured data showing all 250 pairs inside a 0.3-wide
band far from zero — is a real API-design decision backed by evidence.

The test suites are not decoration. 73 + 78 tests cover `ties at the k boundary
keep the lowest rows`, `an existing row is never mutated in place`, `a leftover
staging directory does not block compaction`, `the tokenizer lowercases
locale-independently`. The README snippets are executed by `test_readme.py` and
`ReadmeSpec.kt`, so they cannot drift.

Given that baseline, the findings below are narrow. They are real, but none of
them makes the project's central claim false.

---

## Findings

### 1. Kotlin reports a future format version as corruption

`Manifest.decode` deserializes the whole `ManifestWire` *before* checking
`formatVersion`. Python checks the version first, then decodes.

```kotlin
val wire = try {
    WireJson.decodeFromString(ManifestWire.serializer(), raw)   // <- fails first
} catch (exc: SerializationException) {
    throw CorruptIndexException("$path: manifest is not valid JSON ...")
}
if (wire.formatVersion != FORMAT_VERSION) { throw FormatVersionException(...) }
```

A v2 manifest that changes the field set — which is what §9 *defines* a version
bump to be — produces the wrong error. Verified against a `format_version: 2`
manifest:

```
Python:  FormatVersionError: declares format_version 2; this build supports version 1
Kotlin:  CorruptIndexException: manifest is not valid JSON
         (Fields [chunker_id, row_count, live_count, created_at, updated_at]
          are required ... but they were missing)
```

SPEC §2.2 lists "`format_version` is known" as **check 1**, before anything
structural. The existing Kotlin test `open refuses an unknown format version`
passes because it rewrites `"format_version":1` to `99` inside an otherwise
complete v1 manifest — every other field is still present, so deserialization
succeeds and the version check is reached. The test's construction is precisely
what hides the ordering bug.

This matters on the day v2 ships: every v1 Kotlin reader in the field will tell
its user the index is corrupt rather than that the library is too old.

**Fix.** Peek `format_version` before decoding the rest — a two-field
`@Serializable` probe class, or a `JsonObject` lookup — then decode the full wire.

### 2. `searchVector` can exhaust memory on a large `k` (Kotlin)

```kotlin
val heap = PriorityQueue<Hit>(k, WORST_FIRST)
```

`PriorityQueue` allocates its backing array eagerly, so `k` is an allocation
request, not a bound. Verified on a **5-row** index:

```
searchVector(q, Int.MAX_VALUE) -> OutOfMemoryError: Requested array size exceeds VM limit
searchVector(q, 5_000_000)     -> ok, 5 results   (after allocating 5M slots)
```

Python caps first (`limit = min(k, rows.size)`) and returns 5 results. This is
reachable straight from the CLI (`search ... -k 2147483647`), and §8 explicitly
promises that "`k` larger than the live count returns all live rows" — which
Kotlin satisfies only when it survives the allocation.

**Fix.** `PriorityQueue(minOf(k, current.liveCount).coerceAtLeast(1), WORST_FIRST)`.

### 3. The append path is crash-*detecting*, not crash-*safe*

`compact()` is genuinely crash-safe: it builds a complete index in a staging
directory, `fsync`s every file and the directory, then renames. The append path
is not held to the same standard. In both languages, `vectors.f32` and
`docs.jsonl` are appended with a plain write and **no `fsync`**, while
`offsets.bin`, `tombstones.bits` and `manifest.json` each go through
`write_atomic` (write → fsync → rename → fsync parent).

So after `add_all` returns, the manifest is durable and the bulk data may not be.
On power loss the next `open()` hits §2.2 check 4 and refuses the index as
truncated. The docstring justifies the ordering only for the *opposite* failure:

> a crash before it leaves a `vectors.f32` longer than `row_count` implies, which
> SPEC.md §2.2's length check catches on the next open

That reasoning holds when data is durable and the manifest isn't. It does not
cover manifest-durable/data-not, which is the ordering the code actually
produces. The result is not silent corruption — the check fires — but the
outcome is an *unopenable* index rather than one that has merely lost its last
batch.

**Fix.** `fsync` both appended files before writing the manifest. Given
`compact()` already pays exactly this cost, the asymmetry looks unintended.

### 4. Cross-implementation coverage has a tombstone-shaped hole

`tombstones.bits` never crosses the language boundary anywhere in conformance:

- `compare_files` iterates `vectors.f32 docs.jsonl offsets.bin` only.
- `run.sh` never deletes or compacts — no mention of either.
- Neither CLI exposes a `delete` subcommand, so the harness *cannot* produce a
  tombstone file.
- `conformance/fixtures/golden-v1/` contains four files; no `tombstones.bits`.

This is the one file §6 singles out as silently misparseable: "MSB-first produces
a file of the same size that parses without error and disagrees about which rows
are deleted."

**In fairness, this is narrower than it first looks.** Both unit suites pin the
same literal bytes for the same case (rows 0 and 9 → `0x01`, `0x02`), so an
encoder drift would be caught in-language. I also checked the round-trip
empirically — a 20-row index with deletions at `{0,3,7,8,15,19}`:

```
Kotlin: 898108      Python: 898108      (both: 14 live of 20)
```

They agree. What is missing is the *end-to-end* guard: nothing verifies that one
implementation's `tombstones.bits` is read correctly by the other, nor that
`live_count` and search results agree over a tombstoned index across languages.
The encoder is pinned; the reader is pinned only against its own writer.

**Fix.** Add `delete` to both CLIs, then extend `run.sh` to delete a few ids,
compare `tombstones.bits` byte-for-byte, and re-run the query sweep. A golden
fixture containing tombstones would also close the v1 regression gap.

### 5. The replay fixture uses two different decimal→float32 paths

Appendix B justifies the replay fixture as a *stronger* byte-identity check than
the hashing embedder, because "decimal parsing, float32 narrowing and summation
order all have to agree." The two implementations parse those decimals
differently:

| | declared type | conversion |
|---|---|---|
| Kotlin | `vector: List<Float>` | `Float.parseFloat` — decimal → binary32, one rounding |
| Python | `vector: list[float]` → `np.asarray(..., np.float32)` | decimal → binary64 → binary32, **two roundings** |

Double rounding is not always equal to single rounding, so the agreement the spec
claims to *prove* is not established by construction.

I measured the actual risk rather than assuming it. All **26,880** values in the
committed fixture agree under both paths. Divergence requires a decimal landing
within ~2⁻⁵³ of a float32 midpoint, roughly a 2⁻²⁸ chance per value; 40M random
6-significant-digit decimals in the fixture's magnitude range produced zero
divergences, and a hand-constructed counterexample needed 27 significant digits
(`1.00000005960464477539062501` → `1.0000001` direct vs `1.0` via double). For a
768×35 fixture that is roughly **1 in 10⁴ per regeneration**.

So: low probability, currently passing, and never going to be caught by CI if it
does happen — the failure would look like an unreproducible byte-identity break.
The fix is one word (`List<Double>` in `FixtureRecordWire`, then narrow, matching
Python), which is cheap enough that a latent 10⁻⁴ is not worth carrying in a
project whose entire premise is bit-exactness.

### 6. §5's ordering invariant is never validated

SPEC §5 says offsets "entries strictly increase". Both readers check only the
length, `(row_count + 1) * 8`. Nothing checks monotonicity, and the two languages
degrade differently on a file that passes the length check:

- Python: `handle.read(negative)` reads to EOF → JSON decode failure →
  `CorruptIndexError`. Loud, but by luck rather than by design.
- Kotlin: `ByteArray((end - start).toInt())` → `NegativeArraySizeException`, a raw
  JVM error that is neither `SimpleSemanticException` nor
  `IllegalArgumentException` and so escapes the CLI's handlers as a stack trace.

Given the care in §2.2's other six checks, this one reads like an oversight. A
single ascending pass in `readOffsets` / `read_offsets` closes it.

---

## Smaller notes

**Python's `_closed` flag is dead.** It is set in `close()` and never read.
Searching a closed index diverges:

```
Python: IndexError: index 0 is out of bounds for axis 0 with size 0
Kotlin: SimpleSemanticException: index is closed
```

The Kotlin message is the right one; Python leaks a NumPy internal. The flag is
already there — it just needs a guard.

**Kotlin's CLI argument parser reads past the end.** `Options.parse` does
`args[++i]` with no bounds check, so a trailing `--dimension` throws
`ArrayIndexOutOfBoundsException` — which `main` catches as neither
`SimpleSemanticException` nor `IllegalArgumentException`, so the user gets a stack
trace. (`--dimension abc` is fine: `NumberFormatException` *is* an
`IllegalArgumentException`.) Python's argparse handles both cleanly.

**`@SemanticIndexed` packs its sort key.** `annotation.order * 1000 + position`
collides past 1000 fields, or with a negative `order`. `compareBy({ it.order },
{ it.position })` is exact and no harder to read. Contrived in practice, but it's
an unnecessary numeric assumption in code whose job is pinning order.

**`httpx` is a hard dependency for every Python install.** `gemini.py` opens
with:

> In its own module so importing `simple_semantic` does not pull in an HTTP client.

The lazy import achieves that at *import* time, but `pyproject.toml` lists
`httpx` in core `dependencies`, so every install still gets it. Kotlin's
`GeminiEmbedder` uses the JDK's `java.net.http.HttpClient` and adds no dependency
at all. Moving `httpx` to an optional `gemini` extra would make the packaging
match the stated intent — and match Kotlin.

**Document fetch opens the file per result.** Both `documentAt` and
`_document_at` open `docs.jsonl` fresh for every hit, so a `k=10` search is 10
`open()` syscalls. Correct, and irrelevant next to the brute-force scan, but a
single held handle is the obvious form.

---

## Overall

The engineering here is well above average, and the reasons are structural rather
than stylistic: a normative spec that argues for its own decisions, two
independent implementations that keep each other honest, and a conformance suite
that checks bytes rather than behaviour. The comment `# Step 3 is the one that
would be easy to quietly weaken. Do not.` is the whole project's posture in one
line.

The findings cluster in a telling way. Four of the six — the version-check
ordering, the missing tombstone conformance, the two decimal→float32 paths, and
the unvalidated offset ordering — are all cases where **one implementation is
right and the other is right by coincidence**, and the conformance suite happens
not to look there. That is the characteristic failure mode of a two-implementation
design: the suite proves agreement on the paths it exercises, and those paths are
the ones both authors thought about. The gaps are in error handling, in the
optional file, and in a numeric conversion that agrees 99.99% of the time.

None of this is urgent. The most valuable single change is **#1** — it costs a
few lines and determines whether the v2 migration is smooth or generates a wave
of spurious "corrupt index" reports. **#4** is the one that would most strengthen
the project's central claim, since it closes the last file that has never been
compared across the boundary. **#3** is worth deciding deliberately either way:
the current behaviour may well be an acceptable trade, but the docstring argues
for a guarantee the code doesn't provide, and that gap between stated and actual
is the kind of thing this project otherwise never tolerates.
