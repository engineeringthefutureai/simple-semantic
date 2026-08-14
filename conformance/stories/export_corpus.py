#!/usr/bin/env python3
"""Export the story corpus in the shapes the CLI takes.

The CLI wants JSONL documents and a newline-separated query list. Ids, filenames
and prompts come from the generated fixture, so this needs no YAML library and
runs under the project's own venv.

    python conformance/stories/export_corpus.py \
        --documents out/stories.jsonl --queries out/story-queries.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

STORIES_DIR = Path(__file__).parent
FIXTURE = STORIES_DIR.parent / "fixtures" / "story-embeddings-v1.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, required=True, help="JSONL output path")
    parser.add_argument("--queries", type=Path, required=True, help="query list output path")
    args = parser.parse_args()

    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))

    args.documents.parent.mkdir(parents=True, exist_ok=True)
    with args.documents.open("w", encoding="utf-8") as handle:
        for record in fixture["documents"]:
            text = (STORIES_DIR / record["file"]).read_text(encoding="utf-8")
            handle.write(json.dumps({"id": record["id"], "text": text}, ensure_ascii=False) + "\n")

    # A blank query is skipped by search on both sides, and an empty line here
    # would be read as one.
    prompts = [record["prompt"] for record in fixture["queries"] if record["prompt"].strip()]
    args.queries.write_text("\n".join(prompts) + "\n", encoding="utf-8")

    print(f"ok    exported {len(fixture['documents'])} stories and {len(prompts)} queries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
