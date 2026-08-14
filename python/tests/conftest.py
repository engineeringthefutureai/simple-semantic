from __future__ import annotations

from pathlib import Path

import pytest

from simple_semantic import HashingEmbedder, SemanticIndex


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dimension=64, seed=0)


@pytest.fixture
def index(tmp_path: Path, embedder: HashingEmbedder):
    """A fresh index per test.

    Per-test isolation is not incidental. The `simple-fts` specs shared one
    mutable index across cases, which made every failure ambiguous: a broken
    delete showed up as a failure in an unrelated search test three cases
    later.
    """
    handle = SemanticIndex.create(tmp_path / "index", embedder)
    yield handle
    handle.close()
