"""Loading the story corpus from the generated fixture.

`build_fixture.py` is the only thing that parses the YAML in
`conformance/stories/`; it records ids, filenames, prompts and targets alongside
the vectors, so this reads JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STORIES_DIR = REPO_ROOT / "conformance" / "stories"
FIXTURE = REPO_ROOT / "conformance" / "fixtures" / "story-embeddings-v1.json"


@dataclass(frozen=True)
class Story:
    id: str
    text: str


@dataclass(frozen=True)
class Query:
    prompt: str
    target: str

    @property
    def is_negative(self) -> bool:
        """No correct answer by construction — tax returns, sourdough, qubits."""
        return not self.target


def _fixture() -> dict[str, list[dict[str, str]]]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def load_stories() -> list[Story]:
    return [
        Story(
            id=record["id"],
            text=(STORIES_DIR / record["file"]).read_text(encoding="utf-8"),
        )
        for record in _fixture()["documents"]
    ]


def load_queries() -> list[Query]:
    return [
        Query(prompt=record["prompt"], target=record["target"]) for record in _fixture()["queries"]
    ]
