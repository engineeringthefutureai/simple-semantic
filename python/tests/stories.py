"""Loading the story corpus. Shared by the test suite and nothing else.

The corpus lives in ``conformance/stories/``: ten AI-generated stories across
genres, twenty targeted queries, and five negative controls. The YAML there is
the helper's format; the vectors this suite uses come from the generated JSON
fixture, so the tests need no YAML dependency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STORIES_DIR = REPO_ROOT / "conformance" / "stories"
FIXTURE = REPO_ROOT / "conformance" / "fixtures" / "story-embeddings-v1.json"

#: Queries in this category have no correct answer by construction — they are
#: about corporate tax returns, sourdough, quantum computing and bicycle brakes.
NEGATIVE_CATEGORY = "irrelevant_queries"


@dataclass(frozen=True)
class Story:
    id: str
    genre: str
    title: str
    text: str
    word_count: int


@dataclass(frozen=True)
class Query:
    prompt: str
    target: str
    category: str

    @property
    def is_negative(self) -> bool:
        return self.category == NEGATIVE_CATEGORY


def _parse_scalar(value: str) -> str | int:
    value = value.strip()
    if value.startswith('"') and value.endswith('"'):
        return json.loads(value)
    if value.lstrip("-").isdigit():
        return int(value)
    return value


def load_stories() -> list[Story]:
    """Read metadata.yaml without a YAML dependency.

    The file is a flat, machine-written subset of YAML — ``---`` separators and
    ``key: value`` lines, no nesting, no anchors — so a twenty-line reader is
    honest here and keeps the test suite's dependencies equal to the library's.
    If metadata.yaml ever grows real YAML, this should become a dependency
    rather than a cleverer parser.
    """
    stories: list[Story] = []
    current: dict[str, str | int] = {}
    for line in (STORIES_DIR / "metadata.yaml").read_text(encoding="utf-8").splitlines():
        if line.startswith("---"):
            if current:
                stories.append(_to_story(current))
            current = {}
            continue
        if not line.strip() or line.startswith("embedding:"):
            continue
        key, _, value = line.partition(":")
        current[key.strip()] = _parse_scalar(value)
    if current:
        stories.append(_to_story(current))
    return stories


def _to_story(entry: dict[str, str | int]) -> Story:
    return Story(
        id=str(entry["id"]),
        genre=str(entry["genre"]),
        title=str(entry["title"]),
        text=(STORIES_DIR / str(entry["file"])).read_text(encoding="utf-8"),
        word_count=int(entry["word_count"]),
    )


def load_queries() -> list[Query]:
    queries: list[Query] = []
    category = ""
    prompt: str | None = None
    target = ""
    # The category is captured when a prompt *starts*, not when it is flushed.
    # Flushing on the next prompt would attribute the last entry of each block
    # to the following block's heading.
    prompt_category = ""

    def flush() -> None:
        if prompt is not None:
            queries.append(Query(prompt, target, prompt_category))

    for line in (STORIES_DIR / "queries.yaml").read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if not line.startswith(" "):
            category = line.rstrip(":").strip()
            continue
        stripped = line.strip()
        if stripped.startswith("- prompt:"):
            flush()
            prompt = str(_parse_scalar(stripped[len("- prompt:") :]))
            prompt_category = category
            target = ""
        elif stripped.startswith("target_story:"):
            target = str(_parse_scalar(stripped[len("target_story:") :]))
    flush()
    return queries
