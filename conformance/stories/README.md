# Conformance Stories

This directory contains 10 benchmark stories across diverse fiction genres.

> [!IMPORTANT]
> **AI Generation Notice**: All stories in this directory are **100% AI one-shot generated** with no human seed or human editing. They are intended **strictly for testing and benchmarking purposes only** (e.g., verifying semantic search retrieval, indexing conformance, and vector query performance across diverse prose styles and domain vocabularies).

---

## Batch Embedding Script

A Python script [generate_embeddings.py](generate_embeddings.py) is provided to generate embeddings for all 10 story files and 25 queries using Google's official `google-genai` SDK.

### Setup & Execution:

```bash
pip install google-genai pyyaml numpy
export GEMINI_API_KEY="your-gemini-api-key"
python conformance/stories/generate_embeddings.py
```

Defaults to `gemini-embedding-001` at 768 dimensions, which is what produced
the committed vectors. Override with `GEMINI_EMBED_MODEL` / `GEMINI_EMBED_DIM`,
but note that the model identity is recorded per entry as `embedder_id` and
the whole corpus must come from one model — `build_fixture.py` refuses a mix.

At 768 dimensions `gemini-embedding-001` returns **unnormalized** vectors
(norms near 0.59); only its default 3072-dimension output is pre-normalized.
That is expected and is left as-is: the index L2-normalizes every row at
write time, and keeping the recording unnormalized is what lets the test
suite prove it.

---

## Matrix Score Inspection Script

A debug/inspection script [inspect_scores.py](inspect_scores.py) multiplies the query embedding matrix by the story embedding matrix ($Q \times S^T$) and outputs a formatted table of similarity scores.

### Modes:
- **Raw Cosine Similarity**:
  ```bash
  python conformance/stories/inspect_scores.py
  ```
- **Min-Max Normalized [0.00 to 1.00]** (highlights contrast against the
  background noise floor). Read it as contrast only: min-max forces the best
  cell to exactly 1.00 and the worst to 0.00, which makes an uncalibrated
  0.30-wide band look like a calibrated score. The library deliberately
  exposes no absolute threshold, for the reason this view makes vivid:
  ```bash
  python conformance/stories/inspect_scores.py --minmax
  ```
- **Z-Score Standardized** (std deviations from corpus baseline mean):
  ```bash
  python conformance/stories/inspect_scores.py --zscore
  ```

---

## Fixture used by the test suites

`build_fixture.py` converts the YAML above into
[`../fixtures/story-embeddings-v1.json`](../fixtures/story-embeddings-v1.json),
a language-neutral recording keyed by `sha256(text)`:

```bash
python conformance/stories/build_fixture.py
```

That file is what `ReplayEmbedder` reads in both implementations, so the
Python and Kotlin suites make identical retrieval assertions offline, and
`conformance/run.sh` can prove byte-identity from real embeddings rather than
only from the deterministic hashing embedder. Documents and queries are kept
in separate maps because they were embedded with different task types.

The scripts in this directory are **helpers**, not library code. The library
itself never reads YAML and never calls the `google-genai` SDK.

---

## Hypothetical Search Queries

Hypothetical semantic search queries are stored in [queries.yaml](queries.yaml). They test concept retrieval without relying on literal character names or story keywords:
- **5 short queries** (up to 3 words)
- **5 medium queries** (up to 5 words)
- **10 long queries** (up to 10 words)
- **5 irrelevant queries** (negative controls of various lengths, e.g. corporate tax returns, baking sourdough bread, quantum computing, hydraulic bike brakes)

---

## Metadata

The metadata for all stories (genre, title, number, file path, word count, character count) is consolidated in [metadata.yaml](metadata.yaml) as `---` separated multi-document YAML sections.

```yaml
---
id: story-01
number: 1
genre: "Cyberpunk / Science Fiction"
title: "The Unwritten Sun"
file: "the-unwritten-sun.md"
word_count: 749
char_count: 4593
embedding: [0.012345, -0.045678, ...]
```

---

## Story Index

| # | Genre | Story Title | Content File |
|---|---|---|---|
| 1 | Cyberpunk / Science Fiction | *The Unwritten Sun* | [the-unwritten-sun.md](the-unwritten-sun.md) |
| 2 | Historical Fiction (Renaissance) | *The Venice Nightingale* | [the-venice-nightingale.md](the-venice-nightingale.md) |
| 3 | Magical Realism | *The Tide of Whispers* | [the-tide-of-whispers.md](the-tide-of-whispers.md) |
| 4 | Mystery / Noir | *The Reverse Escapement* | [the-reverse-escapement.md](the-reverse-escapement.md) |
| 5 | High Fantasy | *The Living Meridian* | [the-living-meridian.md](the-living-meridian.md) |
| 6 | Psychological Thriller | *The Echoes in Room 4B* | [the-echoes-in-room-4b.md](the-echoes-in-room-4b.md) |
| 7 | Slice-of-Life / Domestic Drama | *The Prairie Beacon* | [the-prairie-beacon.md](the-prairie-beacon.md) |
| 8 | Post-Apocalyptic / Dystopian | *The Unmetered Moss* | [the-unmetered-moss.md](the-unmetered-moss.md) |
| 9 | Gothic Horror | *The Varnish of Penitence* | [the-varnish-of-penitence.md](the-varnish-of-penitence.md) |
| 10 | Satirical Comedy | *The Department of Micro-Frustrations* | [the-department-of-micro-frustrations.md](the-department-of-micro-frustrations.md) |
