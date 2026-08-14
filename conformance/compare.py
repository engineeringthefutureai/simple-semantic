#!/usr/bin/env python3
"""Compare two conformance runs.

Ranked id lists must match exactly; scores may diverge by up to 1e-6, because
SPEC.md §8 leaves the scoring reduction order free while §3.1 pins
normalization's.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCORE_TOLERANCE = 1e-6


def load(path: Path) -> list[list[dict[str, object]]]:
    """Read a results file: blank-line-separated blocks, one per query."""
    blocks: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            blocks.append(current)
            current = []
            continue
        current.append(json.loads(line))
    blocks.append(current)
    return blocks


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print("usage: compare.py <left.jsonl> <right.jsonl> <label>", file=sys.stderr)
        return 2

    left_path, right_path, label = Path(argv[1]), Path(argv[2]), argv[3]
    left, right = load(left_path), load(right_path)

    failures: list[str] = []
    if len(left) != len(right):
        failures.append(f"{len(left)} query blocks on the left, {len(right)} on the right")

    for query_index, (left_block, right_block) in enumerate(zip(left, right, strict=False)):
        left_ids = [row["id"] for row in left_block]
        right_ids = [row["id"] for row in right_block]
        if left_ids != right_ids:
            failures.append(
                f"query {query_index}: ranked ids differ\n"
                f"    left:  {left_ids}\n"
                f"    right: {right_ids}"
            )
            continue
        for rank, (left_row, right_row) in enumerate(zip(left_block, right_block, strict=True)):
            delta = abs(float(str(left_row["score"])) - float(str(right_row["score"])))
            if delta > SCORE_TOLERANCE:
                failures.append(
                    f"query {query_index} rank {rank} ({left_row['id']}): "
                    f"scores differ by {delta:.3e} "
                    f"({left_row['score']} vs {right_row['score']})"
                )

    scored = sum(len(block) for block in left)
    if failures:
        print(f"FAIL  {label}")
        for failure in failures:
            print(f"      {failure}")
        return 1

    print(f"ok    {label} ({len(left)} queries, {scored} results)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
