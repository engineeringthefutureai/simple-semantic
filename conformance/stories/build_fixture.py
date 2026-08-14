#!/usr/bin/env python3
"""Turn the helper YAML into a language-neutral fixture both implementations read.

The YAML files in this directory are the helper's working format: convenient to
diff, convenient to hand-edit, and readable only with a YAML library. The
library's core takes numpy and an HTTP client and nothing else (see CLAUDE.md),
and the Kotlin implementation has no YAML parser at all — so the tested artifact
is JSON, generated from the YAML by this script.

Output: ``conformance/fixtures/story-embeddings-v1.json``

    {
      "fixture_version": 1,
      "embedder_id": "gemini-embedding-001@768",
      "dimension": 768,
      "documents": [{"key": "<sha256 of the embedded text>", "vector": [...]}],
      "queries":   [{"key": "<sha256 of the embedded text>", "vector": [...]}]
    }

Vectors are stored **exactly as the model returned them**, unnormalized.
``gemini-embedding-001`` pre-normalizes only its default 3072-dimension output;
at 768 the norms come back around 0.59. Keeping them that way is deliberate: it
means the test suite exercises SPEC.md §3.1 write-time normalization against
real unnormalized input rather than against a vector that was already unit-norm.

Documents and queries are kept in **separate maps** on purpose. They were
embedded with different task types (``RETRIEVAL_DOCUMENT`` versus
``RETRIEVAL_QUERY``), so the same text embedded both ways is two different
vectors. A single flat map would quietly serve one where the other was meant,
which is exactly the asymmetry the ``Embedder`` interface exists to preserve.

Usage:
    python conformance/stories/build_fixture.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    print("Error: 'pyyaml' is not installed. Install it with: pip install pyyaml")
    sys.exit(1)

FIXTURE_VERSION = 1
QUERY_CATEGORIES = ["short_queries", "medium_queries", "long_queries", "irrelevant_queries"]


def content_key(text: str) -> str:
    """Key a recorded vector by the sha256 of the exact text that produced it.

    Not by document id: a replay embedder is given text, not ids, and keying on
    text means an edited story stops matching its stale vector instead of
    silently keeping it.
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
            records.append(
                {
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

    # One embedder for the whole fixture, or the recorded vectors are not
    # comparable with each other. This is SPEC.md §2.1 applied to the fixture
    # itself: mixing two models here would produce meaningless rankings with no
    # error anywhere downstream.
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
        # Stated so a reader does not have to measure it to learn that these
        # vectors are not unit-norm, and that this is expected.
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
