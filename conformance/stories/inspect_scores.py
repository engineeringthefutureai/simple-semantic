#!/usr/bin/env python3
"""Debug & Inspection Script: Story x Query Similarity Score Matrix.

Loads story embeddings from metadata.yaml and query embeddings from queries.yaml,
computes the dot product / cosine similarity matrix (Q_norm @ S_norm.T), and prints a
formatted matrix table for manual inspection.

Features:
- Supports raw cosine scores or --minmax / --zscore normalized scores for enhanced contrast.
- Prompts labeled P01-P25 and stories labeled S01-S10.
- Target story printed in S01-S10 notation.
- Added 'Top 3 Matches' column showing top 3 most relevant stories sorted by similarity.
- Native fixed-width column formatting.
- Full prompt and story legends printed to stderr.

Usage:
    python conformance/stories/inspect_scores.py
    python conformance/stories/inspect_scores.py --minmax
    python conformance/stories/inspect_scores.py --zscore
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser(description="Inspect query x story similarity score matrix.")
    parser.add_argument(
        "--minmax",
        action="store_true",
        help="Apply min-max normalization across the matrix [0.00 to 1.00] to highlight relative contrast.",
    )
    parser.add_argument(
        "--zscore",
        action="store_true",
        help="Apply z-score standardization (mean=0, std=1) to measure standard deviations from baseline noise.",
    )
    args = parser.parse_args()

    stories_dir = Path(__file__).parent
    metadata_path = stories_dir / "metadata.yaml"
    queries_path = stories_dir / "queries.yaml"

    if not metadata_path.exists() or not queries_path.exists():
        print(f"Error: Missing {metadata_path} or {queries_path}", file=sys.stderr)
        sys.exit(1)

    # 1. Load Story Metadata & Embeddings
    with open(metadata_path, "r", encoding="utf-8") as f:
        story_docs = list(yaml.safe_load_all(f))
    story_entries = [d for d in story_docs if d and isinstance(d, dict)]

    story_ids = []
    story_titles = []
    story_vecs = []

    for entry in story_entries:
        if "embedding" not in entry or not entry["embedding"]:
            print(
                f"Error: Story '{entry.get('title')}' ({entry.get('id')}) has no embedding.\n"
                f"Please run 'python conformance/stories/generate_embeddings.py' first.",
                file=sys.stderr,
            )
            sys.exit(1)
        story_ids.append(entry["id"])
        story_titles.append(entry.get("title", entry["id"]))
        story_vecs.append(entry["embedding"])

    # Mapping story_id -> S01..S10 notation
    id_to_code = {sid: f"S{i+1:02d}" for i, sid in enumerate(story_ids)}

    # Story Matrix S: (10, D)
    S = np.array(story_vecs, dtype=np.float32)
    norms_S = np.linalg.norm(S, axis=1, keepdims=True)
    norms_S[norms_S == 0] = 1.0
    S_norm = S / norms_S

    # 2. Load Query Metadata & Embeddings
    with open(queries_path, "r", encoding="utf-8") as f:
        queries_data = yaml.safe_load(f)

    query_prompts = []
    target_stories = []
    query_categories = []
    query_vecs = []

    categories = ["short_queries", "medium_queries", "long_queries", "irrelevant_queries"]
    for cat in categories:
        if cat in queries_data:
            for item in queries_data[cat]:
                if "embedding" not in item or not item["embedding"]:
                    print(
                        f"Error: Query '{item.get('prompt')}' has no embedding.\n"
                        f"Please run 'python conformance/stories/generate_embeddings.py' first.",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                query_prompts.append(item["prompt"])
                target_stories.append(item.get("target_story", ""))
                query_categories.append(cat)
                query_vecs.append(item["embedding"])

    # Query Matrix Q: (25, D)
    Q = np.array(query_vecs, dtype=np.float32)
    norms_Q = np.linalg.norm(Q, axis=1, keepdims=True)
    norms_Q[norms_Q == 0] = 1.0
    Q_norm = Q / norms_Q

    # 3. Compute Similarity Scores Matrix: Q_norm @ S_norm.T (25 x 10)
    raw_scores = Q_norm @ S_norm.T

    # 4. Optional Calibration / Normalization
    mode_name = "RAW COSINE SIMILARITY"
    if args.minmax:
        s_min, s_max = float(np.min(raw_scores)), float(np.max(raw_scores))
        display_scores = (raw_scores - s_min) / (s_max - s_min) if s_max > s_min else raw_scores
        mode_name = f"MIN-MAX NORMALIZED [0.00 to 1.00] (raw min={s_min:+.4f}, max={s_max:+.4f})"
    elif args.zscore:
        mean, std = float(np.mean(raw_scores)), float(np.std(raw_scores))
        display_scores = (raw_scores - mean) / std if std > 0 else raw_scores
        mode_name = f"Z-SCORE STANDARDIZED (raw mean={mean:+.4f}, std={std:.4f})"
    else:
        display_scores = raw_scores

    # 5. Output Formatted Table
    short_headers = [f"{f'S{i+1:02d}':>8}" for i in range(len(story_ids))]

    print(f"\n==================== SIMILARITY SCORE MATRIX ({mode_name}) ====================")

    # Table Header
    header_cols = [f"{'Prompt':<6}", f"{'Target':<6}", f"{'Top 3 Matches':<15}"] + short_headers
    print(" ".join(header_cols))

    for idx, prompt in enumerate(query_prompts):
        prompt_code = f"P{idx+1:02d}"
        target_raw = target_stories[idx]
        target_code = id_to_code.get(target_raw, target_raw)

        # Top 3 based on raw cosine scores
        top3_indices = np.argsort(raw_scores[idx])[::-1][:3]
        top3_str = ", ".join(f"S{j+1:02d}" for j in top3_indices)

        row_scores = [f"{display_scores[idx, j]:>+8.4f}" for j in range(len(story_ids))]
        print(" ".join([f"{prompt_code:<6}", f"{target_code:<6}", f"{top3_str:<15}"] + row_scores))

    # Print Legends to stderr so stdout table stays clean
    print("\n==================== PROMPT LEGEND ====================", file=sys.stderr)
    for idx, (prompt, target, cat) in enumerate(zip(query_prompts, target_stories, query_categories), 1):
        target_code = id_to_code.get(target, target)
        print(f"P{idx:02d}: \"{prompt}\" (target: {target_code}, category: {cat})", file=sys.stderr)

    print("\n==================== STORY LEGEND =====================", file=sys.stderr)
    for i, (sid, title) in enumerate(zip(story_ids, story_titles), 1):
        print(f"S{i:02d}: [{sid}] {title}", file=sys.stderr)


if __name__ == "__main__":
    main()
