#!/usr/bin/env python3
"""Compare two manifests on everything except their timestamps.

`manifest.json` cannot be byte-identical across two writers: `created_at` and
`updated_at` are wall-clock. Every other field must agree exactly, so those two
are excluded by name rather than the whole file being skipped.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

VOLATILE = {"created_at", "updated_at"}


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: compare_manifests.py <left.json> <right.json>", file=sys.stderr)
        return 2

    left = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    right = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
    stable_left = {key: value for key, value in left.items() if key not in VOLATILE}
    stable_right = {key: value for key, value in right.items() if key not in VOLATILE}

    if stable_left != stable_right:
        print("FAIL  manifest.json differs beyond its timestamps")
        print(f"      left:  {stable_left}")
        print(f"      right: {stable_right}")
        return 1

    print("ok    manifest.json agrees on every non-timestamp field")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
