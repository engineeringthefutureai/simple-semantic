#!/usr/bin/env python3
"""Turn the helper YAML into a language-neutral fixture both implementations read.

The YAML in this directory is the helper's working format. The library takes
numpy and an HTTP client and nothing else, and Kotlin has no YAML parser, so the
tested artifact is JSON, generated from the YAML by this script.

Output: ``conformance/fixtures/story-embeddings-v1.json``

    {
      "fixture_version": 1,
      "embedder_id": "gemini-embedding-001@768",
      "dimension": 768,
      "documents": [{"id": ..., "file": ..., "key": "<sha256>", "vector": [...]}],
      "queries":   [{"prompt": ..., "target": ..., "key": "<sha256>", "vector": [...]}]
    }

Carries the corpus metadata as well as the vectors, so that build_fixture.py is
the only thing in the repo that parses the YAML. Everything downstream — both
test suites and the conformance harness — reads this JSON.

Vectors are stored exactly as the model returned them, unnormalized: at 768
dimensions ``gemini-embedding-001`` norms come back around 0.59, which is what
exercises SPEC.md §3.1 against real input.

Documents and queries are separate maps: they were embedded with different task
types, so the same text embedded both ways is two different vectors.

Usage:
    uv run --script conformance/stories/build_fixture.py
"""

# /// script
# requires-python = ">=3.11"
# dependencies = ["pyyaml"]
# ///

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

FIXTURE_VERSION = 1
QUERY_CATEGORIES = ["short_queries", "medium_queries", "long_queries", "irrelevant_queries"]


def content_key(text: str) -> str:
    """Key a recorded vector by the sha256 of the exact text that produced it.

    Not by document id: an edited story then stops matching its stale vector.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_stories(stories_dir: Path) -> tuple[list[dict[str, Any]], set[str]]:
    entries = [
        doc
        for doc in yaml.safe_load_all((stories_dir / "metadata.yaml").read_text(encoding="utf-8"))
        if doc and isinstance(doc, dict)
    ]
    records: list[dict[str, Any]] = []
    embedder_ids: set[str] = set()
    for entry in entries:
        if "embedding" not in entry or not entry["embedding"]:
            raise SystemExit(
                f"story {entry.get('id')!r} has no embedding; run generate_embeddings.py"
            )
        text = (stories_dir / entry["file"]).read_text(encoding="utf-8")
        embedder_ids.add(str(entry.get("embedder_id", "")))
        records.append(
            {
                "id": entry["id"],
                "file": entry["file"],
                "key": content_key(text),
                "vector": [float(v) for v in entry["embedding"]],
            }
        )
    return records, embedder_ids


def load_queries(stories_dir: Path) -> tuple[list[dict[str, Any]], set[str]]:
    data = yaml.safe_load((stories_dir / "queries.yaml").read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    embedder_ids: set[str] = set()
    for category in QUERY_CATEGORIES:
        for item in data.get(category, []):
            if "embedding" not in item or not item["embedding"]:
                raise SystemExit(f"query {item.get('prompt')!r} has no embedding")
            embedder_ids.add(str(item.get("embedder_id", "")))
            target = str(item.get("target_story") or "")
            records.append(
                {
                    "prompt": item["prompt"],
                    # Empty target marks a negative control: no correct answer.
                    "target": "" if target == "none" else target,
                    "key": content_key(item["prompt"]),
                    "vector": [float(v) for v in item["embedding"]],
                }
            )
    return records, embedder_ids


def main() -> int:
    stories_dir = Path(__file__).parent
    fixture_path = stories_dir.parent / "fixtures" / "story-embeddings-v1.json"

    documents, document_ids = load_stories(stories_dir)
    queries, query_ids = load_queries(stories_dir)

    # One embedder for the whole fixture, or the vectors are not comparable.
    # SPEC.md §2.1 applied to the fixture itself.
    embedder_ids = {value for value in document_ids | query_ids if value}
    if len(embedder_ids) != 1:
        raise SystemExit(
            f"expected exactly one embedder_id across the corpus, found {sorted(embedder_ids)}. "
            f"Re-run generate_embeddings.py so documents and queries come from one model."
        )
    embedder_id = embedder_ids.pop()

    dimensions = {len(record["vector"]) for record in documents + queries}
    if len(dimensions) != 1:
        raise SystemExit(f"vectors disagree on dimension: {sorted(dimensions)}")
    dimension = dimensions.pop()

    if not embedder_id.endswith(f"@{dimension}"):
        raise SystemExit(
            f"embedder_id {embedder_id!r} does not match the observed dimension {dimension}"
        )

    duplicate_keys = len(documents) - len({record["key"] for record in documents})
    if duplicate_keys:
        raise SystemExit(f"{duplicate_keys} stories share identical text; keys would collide")

    fixture = {
        "fixture_version": FIXTURE_VERSION,
        "embedder_id": embedder_id,
        "dimension": dimension,
        # Stated so a reader need not measure it.
        "normalized": False,
        "documents": documents,
        "queries": queries,
    }

    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    fixture_path.write_text(json.dumps(fixture, ensure_ascii=False), encoding="utf-8")

    size_kb = fixture_path.stat().st_size / 1024
    print(f"wrote {fixture_path} ({size_kb:.0f} KB)")
    print(f"  embedder_id: {embedder_id}")
    print(f"  {len(documents)} documents, {len(queries)} queries, {dimension} dimensions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
