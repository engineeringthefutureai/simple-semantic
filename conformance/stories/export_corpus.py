#!/usr/bin/env python3
"""Export the story corpus in the shapes the CLI and the harness consume.

The helper YAML in this directory is convenient to read and to hand-edit; the
CLI takes JSONL documents and a newline-separated query list. This bridges the
two so that neither the harness nor the CLI grows a YAML dependency.

    python conformance/stories/export_corpus.py \
        --documents out/stories.jsonl --queries out/story-queries.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

STORIES_DIR = Path(__file__).parent


def load_metadata() -> list[dict[str, str]]:
    """Read the flat, machine-written subset of YAML that metadata.yaml is.

    `---` separators and `key: value` lines, no nesting and no anchors. If that
    ever stops being true, this should take a YAML dependency rather than grow a
    cleverer parser.
    """
    entries: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in (STORIES_DIR / "metadata.yaml").read_text(encoding="utf-8").splitlines():
        if line.startswith("---"):
            if current:
                entries.append(current)
            current = {}
            continue
        if not line.strip() or line.startswith(("embedding:", " ")):
            continue
        key, _, value = line.partition(":")
        current[key.strip()] = value.strip().strip('"')
    if current:
        entries.append(current)
    return entries


def load_prompts() -> list[str]:
    prompts: list[str] = []
    for line in (STORIES_DIR / "queries.yaml").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("- prompt:"):
            prompts.append(json.loads(stripped[len("- prompt:") :].strip()))
    return prompts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, required=True, help="JSONL output path")
    parser.add_argument("--queries", type=Path, required=True, help="query list output path")
    args = parser.parse_args()

    entries = load_metadata()
    args.documents.parent.mkdir(parents=True, exist_ok=True)
    with args.documents.open("w", encoding="utf-8") as handle:
        for entry in entries:
            text = (STORIES_DIR / entry["file"]).read_text(encoding="utf-8")
            handle.write(
                json.dumps(
                    {"id": entry["id"], "text": text, "meta": {"genre": entry["genre"]}},
                    ensure_ascii=False,
                )
                + "\n"
            )

    prompts = load_prompts()
    # A blank query is skipped by search on both sides, and an empty line here
    # would be read as one — so drop them rather than compare two empty results.
    prompts = [prompt for prompt in prompts if prompt.strip()]
    args.queries.write_text("\n".join(prompts) + "\n", encoding="utf-8")

    print(f"ok    exported {len(entries)} stories and {len(prompts)} recorded queries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
