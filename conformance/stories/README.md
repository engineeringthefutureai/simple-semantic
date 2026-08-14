# Conformance Stories

This directory contains 10 benchmark stories across diverse fiction genres.

> [!IMPORTANT]
> **AI Generation Notice**: All stories in this directory are **100% AI one-shot generated** with no human seed or human editing. They are intended **strictly for testing and benchmarking purposes only** (e.g., verifying semantic search retrieval, indexing conformance, and vector query performance across diverse prose styles and domain vocabularies).

---

## Batch Embedding Script

A Python script [generate_embeddings.py](file:///home/kiryl/simple-semantic/conformance/stories/generate_embeddings.py) is provided to generate embeddings for all 10 story files and 20 queries using Google's official `google-genai` SDK.

### Setup & Execution:

```bash
pip install google-genai pyyaml numpy
export GEMINI_API_KEY="your-gemini-api-key"
python conformance/stories/generate_embeddings.py
```

---

## Matrix Score Inspection Script

A debug/inspection script [inspect_scores.py](file:///home/kiryl/simple-semantic/conformance/stories/inspect_scores.py) multiplies the query embedding matrix by the story embedding matrix ($Q \times S^T$) and outputs a formatted table of similarity scores directly:

```bash
python conformance/stories/inspect_scores.py
```

---

## Hypothetical Search Queries

Hypothetical semantic search queries are stored in [queries.yaml](file:///home/kiryl/simple-semantic/conformance/stories/queries.yaml). They are designed for concept retrieval without relying on literal character names or story keywords:
- **5 short queries** (up to 3 words)
- **5 medium queries** (up to 5 words)
- **10 long queries** (up to 10 words)

---

## Metadata

The metadata for all stories (genre, title, number, file path, word count, character count) is consolidated in [metadata.yaml](file:///home/kiryl/simple-semantic/conformance/stories/metadata.yaml) as `---` separated multi-document YAML sections.

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
| 1 | Cyberpunk / Science Fiction | *The Unwritten Sun* | [the-unwritten-sun.md](file:///home/kiryl/simple-semantic/conformance/stories/the-unwritten-sun.md) |
| 2 | Historical Fiction (Renaissance) | *The Venice Nightingale* | [the-venice-nightingale.md](file:///home/kiryl/simple-semantic/conformance/stories/the-venice-nightingale.md) |
| 3 | Magical Realism | *The Tide of Whispers* | [the-tide-of-whispers.md](file:///home/kiryl/simple-semantic/conformance/stories/the-tide-of-whispers.md) |
| 4 | Mystery / Noir | *The Reverse Escapement* | [the-reverse-escapement.md](file:///home/kiryl/simple-semantic/conformance/stories/the-reverse-escapement.md) |
| 5 | High Fantasy | *The Living Meridian* | [the-living-meridian.md](file:///home/kiryl/simple-semantic/conformance/stories/the-living-meridian.md) |
| 6 | Psychological Thriller | *The Echoes in Room 4B* | [the-echoes-in-room-4b.md](file:///home/kiryl/simple-semantic/conformance/stories/the-echoes-in-room-4b.md) |
| 7 | Slice-of-Life / Domestic Drama | *The Prairie Beacon* | [the-prairie-beacon.md](file:///home/kiryl/simple-semantic/conformance/stories/the-prairie-beacon.md) |
| 8 | Post-Apocalyptic / Dystopian | *The Unmetered Moss* | [the-unmetered-moss.md](file:///home/kiryl/simple-semantic/conformance/stories/the-unmetered-moss.md) |
| 9 | Gothic Horror | *The Varnish of Penitence* | [the-varnish-of-penitence.md](file:///home/kiryl/simple-semantic/conformance/stories/the-varnish-of-penitence.md) |
| 10 | Satirical Comedy | *The Department of Micro-Frustrations* | [the-department-of-micro-frustrations.md](file:///home/kiryl/simple-semantic/conformance/stories/the-department-of-micro-frustrations.md) |

Combined collection: [all.md](file:///home/kiryl/simple-semantic/conformance/stories/all.md)
