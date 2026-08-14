#!/usr/bin/env bash
#
# Cross-implementation conformance. The headline test of this project.
#
# The format is the primary artifact; two implementations exist to prove it is
# real. This script is where that claim is checked:
#
#   1. Kotlin writes an index from the shared corpus with HashingEmbedder.
#   2. Python writes its own index from the same corpus.
#   3. vectors.f32, docs.jsonl and offsets.bin must be byte-identical.
#   4. Each implementation runs the query set against *both* indexes.
#   5. Ranked id lists must match exactly; scores within 1e-6.
#   6. Both must still read the committed v1 golden index.
#   7. The same, over the story corpus with real recorded Gemini vectors.
#
# Step 3 is the one that would be easy to quietly weaken. Do not.

set -euo pipefail

# The JVM decodes command-line arguments with sun.jnu.encoding, which follows
# the process locale and is NOT affected by -Dfile.encoding. Under the default
# POSIX locale that is US-ASCII, so a Cyrillic or Japanese query arrives at
# main() as a row of question marks and silently searches for nothing. The
# corpus is deliberately multilingual, so this line is load-bearing.
export LC_ALL="${LC_ALL:-C.UTF-8}"
export LANG="${LANG:-C.UTF-8}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONF="$ROOT/conformance"
OUT="$CONF/out"
CORPUS="$CONF/corpus/corpus.jsonl"
QUERIES="$CONF/corpus/queries.txt"
GOLDEN="$CONF/fixtures/golden-v1"
STORY_FIXTURE="$CONF/fixtures/story-embeddings-v1.json"
STORY_DIR="$CONF/stories"

# The golden fixture is pinned to these, so they are not free parameters.
DIMENSION=64
SEED=0
K=10

rm -rf "$OUT"
mkdir -p "$OUT"

# ---------------------------------------------------------------- toolchains

echo "==> preparing toolchains"

PYTHON="$ROOT/python/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  (cd "$ROOT/python" && uv venv .venv && uv pip install -e ".[dev]" >/dev/null)
fi

KOTLIN_CLI="$ROOT/kotlin/cli/build/install/simple-semantic/bin/simple-semantic"
(cd "$ROOT/kotlin" && ./gradlew --console=plain -q :cli:installDist)

py() { "$PYTHON" -m simple_semantic.cli --dimension "$DIMENSION" --seed "$SEED" "$@"; }
kt() { "$KOTLIN_CLI" --dimension "$DIMENSION" --seed "$SEED" "$@"; }

export PYTHONPATH="$ROOT/python/src${PYTHONPATH:+:$PYTHONPATH}"

# --------------------------------------------------------------- build indexes

echo "==> building indexes from the shared corpus"
kt index "$OUT/kotlin-index" -i "$CORPUS" >/dev/null
py index "$OUT/python-index" -i "$CORPUS" >/dev/null

# ------------------------------------------------------- byte-identity checks

status=0

compare_files() {
  local left="$1" right="$2" label="$3"
  for file in vectors.f32 docs.jsonl offsets.bin; do
    if cmp -s "$left/$file" "$right/$file"; then
      local size
      size=$(wc -c <"$left/$file" | tr -d ' ')
      echo "ok    $label$file identical ($size bytes)"
    else
      echo "FAIL  $label$file differs between implementations"
      cmp "$left/$file" "$right/$file" || true
      status=1
    fi
  done
}

echo "==> comparing files byte for byte"
compare_files "$OUT/kotlin-index" "$OUT/python-index" ""

# manifest.json is deliberately excluded: created_at and updated_at are
# wall-clock timestamps, so it cannot be byte-identical. Everything else in it
# is compared field by field instead.
"$PYTHON" "$CONF/compare_manifests.py" \
  "$OUT/kotlin-index/manifest.json" "$OUT/python-index/manifest.json" || status=1

# --------------------------------------------------------------- query sweeps

# Each implementation queries each index. Four runs, so a disagreement points
# at either the reader or the writer rather than leaving it ambiguous.
run_queries() {
  local runner="$1" index="$2" output="$3" queries="${4:-$QUERIES}"
  : >"$output"
  local first=1
  while IFS= read -r query; do
    if [[ $first -eq 0 ]]; then echo >>"$output"; fi
    first=0
    "$runner" search "$index" "$query" -k "$K" --json >>"$output"
  done <"$queries"
}

echo "==> running the query set four ways"
run_queries kt "$OUT/kotlin-index" "$OUT/kt-on-kt.jsonl"
run_queries py "$OUT/kotlin-index" "$OUT/py-on-kt.jsonl"
run_queries py "$OUT/python-index" "$OUT/py-on-py.jsonl"
run_queries kt "$OUT/python-index" "$OUT/kt-on-py.jsonl"

compare() { "$PYTHON" "$CONF/compare.py" "$@" || status=1; }

# Python reading what Kotlin wrote, and vice versa: the actual interop claim.
compare "$OUT/kt-on-kt.jsonl" "$OUT/py-on-kt.jsonl" "python reads the kotlin index"
compare "$OUT/py-on-py.jsonl" "$OUT/kt-on-py.jsonl" "kotlin reads the python index"
# And the two writers agree with each other, which follows from byte-identity
# but is worth asserting at the level a user actually observes.
compare "$OUT/kt-on-kt.jsonl" "$OUT/py-on-py.jsonl" "both implementations rank identically"

# ----------------------------------------------------------- golden fixture

if [[ -d "$GOLDEN" ]]; then
  echo "==> reading the committed v1 golden index"
  # The regression guard on the format itself: if a change makes this
  # unreadable, the format changed and format_version must change with it.
  run_queries kt "$GOLDEN" "$OUT/kt-on-golden.jsonl"
  run_queries py "$GOLDEN" "$OUT/py-on-golden.jsonl"
  compare "$OUT/kt-on-golden.jsonl" "$OUT/py-on-golden.jsonl" \
    "both implementations agree on the golden index"
  compare "$OUT/kt-on-golden.jsonl" "$CONF/fixtures/golden-v1-expected.jsonl" \
    "the golden index still ranks as it did at v1"
else
  echo "FAIL  no golden fixture at $GOLDEN"
  status=1
fi

# ------------------------------------------------- real-embedding conformance

# Everything above uses HashingEmbedder, whose vectors are small integers before
# normalization. Recorded gemini-embedding-001 output is a strictly harder case
# for byte-identity: 768 arbitrary decimals per row, so any disagreement in
# decimal parsing, float32 narrowing or normalization order shows up here and
# nowhere else in the suite.
if [[ -f "$STORY_FIXTURE" ]]; then
  echo "==> repeating the check with real recorded embeddings"

  "$PYTHON" "$CONF/stories/export_corpus.py" \
    --documents "$OUT/stories.jsonl" --queries "$OUT/story-queries.txt" || status=1

  pyr() { "$PYTHON" -m simple_semantic.cli --embedder replay --fixture "$STORY_FIXTURE" "$@"; }
  ktr() { "$KOTLIN_CLI" --embedder replay --fixture "$STORY_FIXTURE" "$@"; }

  ktr index "$OUT/kotlin-stories" -i "$OUT/stories.jsonl" >/dev/null
  pyr index "$OUT/python-stories" -i "$OUT/stories.jsonl" >/dev/null

  compare_files "$OUT/kotlin-stories" "$OUT/python-stories" "stories "

  # Only the recorded queries can be replayed, so the sweep uses those.
  run_queries ktr "$OUT/kotlin-stories" "$OUT/ktr-on-kt.jsonl" "$OUT/story-queries.txt"
  run_queries pyr "$OUT/python-stories" "$OUT/pyr-on-py.jsonl" "$OUT/story-queries.txt"
  compare "$OUT/ktr-on-kt.jsonl" "$OUT/pyr-on-py.jsonl" \
    "both implementations rank identically on real embeddings"
else
  echo "FAIL  no story fixture at $STORY_FIXTURE"
  echo "      run: python conformance/stories/build_fixture.py"
  status=1
fi

echo
if [[ $status -eq 0 ]]; then
  echo "conformance passed"
else
  echo "conformance FAILED"
fi
exit $status
