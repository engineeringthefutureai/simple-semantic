#!/usr/bin/env python3
"""Batch Embeddings Generator using Google GenAI SDK.

1. Reads story files listed in conformance/stories/metadata.yaml and sends a
   batch request (task_type="RETRIEVAL_DOCUMENT") to generate embeddings for all 10 stories.
   Saves updated stories to metadata.yaml with single-line embedding vectors.

2. Reads search prompts from conformance/stories/queries.yaml and sends a second
   batch request (task_type="RETRIEVAL_QUERY") to generate embeddings for all 20 queries.
   Saves updated queries to queries.yaml with single-line embedding vectors.

Requirements:
    pip install google-genai pyyaml

Usage:
    export GEMINI_API_KEY="your-api-key"
    python conformance/stories/generate_embeddings.py
"""

import json
import os
import sys
from pathlib import Path
import yaml

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Error: 'google-genai' package is not installed.")
    print("Install it with: pip install google-genai pyyaml")
    sys.exit(1)


def format_story_yaml(entry: dict) -> str:
    """Format a story metadata entry as YAML with embedding on a single line."""
    lines = [
        f"id: {entry.get('id', '')}",
        f"number: {entry.get('number', '')}",
        f"genre: {json.dumps(entry.get('genre', ''))}",
        f"title: {json.dumps(entry.get('title', ''))}",
        f"file: {json.dumps(entry.get('file', ''))}",
        f"word_count: {entry.get('word_count', 0)}",
        f"char_count: {entry.get('char_count', 0)}",
    ]
    if "embedding" in entry and entry["embedding"]:
        vec_str = ", ".join(f"{v:.6g}" for v in entry["embedding"])
        lines.append(f"embedding: [{vec_str}]")

    return "\n".join(lines) + "\n"


def format_queries_yaml(data: dict) -> str:
    """Format queries dictionary as YAML with single-line embeddings."""
    lines = [
        "# Hypothetical Semantic Search Queries for Conformance Stories",
        "# Designed to test semantic concept retrieval without literal story keywords or character names.",
        "",
    ]

    for category in ["short_queries", "medium_queries", "long_queries"]:
        if category not in data:
            continue
        lines.append(f"{category}:")
        for item in data[category]:
            lines.append(f"  - prompt: {json.dumps(item['prompt'])}")
            lines.append(f"    target_story: {json.dumps(item['target_story'])}")
            lines.append(f"    word_count: {item['word_count']}")
            if "embedding" in item and item["embedding"]:
                vec_str = ", ".join(f"{v:.6g}" for v in item["embedding"])
                lines.append(f"    embedding: [{vec_str}]")
        lines.append("")

    return "\n".join(lines)


def generate_and_save_all_embeddings():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is not set.")
        print("Please set your API key: export GEMINI_API_KEY='your-api-key'")
        sys.exit(1)

    stories_dir = Path(__file__).parent
    metadata_path = stories_dir / "metadata.yaml"
    queries_path = stories_dir / "queries.yaml"

    if not metadata_path.exists():
        print(f"Error: Metadata file not found at {metadata_path}")
        sys.exit(1)

    if not queries_path.exists():
        print(f"Error: Queries file not found at {queries_path}")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    model_name = os.environ.get("GEMINI_EMBED_MODEL", "text-embedding-004")

    # =========================================================================
    # BATCH 1: Story Documents Embedding (RETRIEVAL_DOCUMENT)
    # =========================================================================
    print("--- STEP 1: Processing Story Documents ---")
    with open(metadata_path, "r", encoding="utf-8") as f:
        documents = list(yaml.safe_load_all(f))

    story_entries = [doc for doc in documents if doc and isinstance(doc, dict)]
    print(f"Loaded metadata for {len(story_entries)} stories.")

    story_texts = []
    story_file_info = []
    for entry in story_entries:
        filename = entry.get("file")
        file_path = stories_dir / filename
        if not file_path.exists():
            print(f"Warning: File {file_path} does not exist, skipping.")
            continue
        content = file_path.read_text(encoding="utf-8")
        story_texts.append(content)
        story_file_info.append(entry)

    if story_texts:
        print(f"Sending Batch Request 1 ({len(story_texts)} documents, task: RETRIEVAL_DOCUMENT)...")
        doc_response = client.models.embed_content(
            model=model_name,
            contents=story_texts,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=768,
            ),
        )
        doc_embeddings = doc_response.embeddings
        print(f"Received {len(doc_embeddings)} document embeddings.")

        for entry, emb in zip(story_file_info, doc_embeddings):
            entry["embedding"] = [float(v) for v in emb.values]

        print(f"Saving updated story metadata to {metadata_path}...")
        with open(metadata_path, "w", encoding="utf-8") as f:
            for entry in story_entries:
                f.write("---\n")
                f.write(format_story_yaml(entry))
        print("Story metadata updated successfully.")

    # =========================================================================
    # BATCH 2: Search Queries Embedding (RETRIEVAL_QUERY)
    # =========================================================================
    print("\n--- STEP 2: Processing Search Queries ---")
    with open(queries_path, "r", encoding="utf-8") as f:
        queries_data = yaml.safe_load(f)

    query_items = []
    query_prompts = []
    for category in ["short_queries", "medium_queries", "long_queries"]:
        if category in queries_data:
            for item in queries_data[category]:
                query_items.append(item)
                query_prompts.append(item["prompt"])

    print(f"Loaded {len(query_prompts)} search query prompts.")

    if query_prompts:
        print(f"Sending Batch Request 2 ({len(query_prompts)} queries, task: RETRIEVAL_QUERY)...")
        query_response = client.models.embed_content(
            model=model_name,
            contents=query_prompts,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=768,
            ),
        )
        query_embeddings = query_response.embeddings
        print(f"Received {len(query_embeddings)} query embeddings.")

        for item, emb in zip(query_items, query_embeddings):
            item["embedding"] = [float(v) for v in emb.values]

        print(f"Saving updated queries to {queries_path}...")
        with open(queries_path, "w", encoding="utf-8") as f:
            f.write(format_queries_yaml(queries_data))
        print("Queries updated successfully.")

    print("\nAll batch embeddings generated and saved successfully!")


if __name__ == "__main__":
    generate_and_save_all_embeddings()
