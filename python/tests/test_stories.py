"""Retrieval quality over the story corpus, with real embeddings.

Every other test in this suite uses ``HashingEmbedder``, which proves the format
is correct and proves nothing about whether search retrieves anything. These
tests run a real ``SemanticIndex`` over ten stories and twenty-five queries whose
vectors came from ``gemini-embedding-001`` — replayed from a fixture, so they
need no API key, no network, and produce identical numbers on every run.

The thresholds below are chosen with slack against the measured values, and each
one names what it is really asserting. They are regression guards, not targets:
a change that improves retrieval should not fail them, and a change that silently
breaks scoring should.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from simple_semantic import (
    CorruptIndexError,
    Document,
    EmbedderMismatchError,
    HashingEmbedder,
    ReplayEmbedder,
    ReplayMissError,
    SemanticIndex,
)
from simple_semantic import format as fmt
from stories import FIXTURE, Query, Story, load_queries, load_stories

pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(),
    reason=f"{FIXTURE} not built; run: uv run --script conformance/stories/build_fixture.py",
)


@pytest.fixture(scope="module")
def corpus() -> tuple[list[Story], list[Query]]:
    return load_stories(), load_queries()


@pytest.fixture
def embedder() -> ReplayEmbedder:
    return ReplayEmbedder.from_file(FIXTURE)


@pytest.fixture
def story_index(tmp_path: Path, embedder: ReplayEmbedder, corpus):
    """A fresh index over all ten stories, per test."""
    stories, _ = corpus
    index = SemanticIndex.create(tmp_path / "stories.index", embedder)
    yield index, stories
    index.close()


async def _fill(index: SemanticIndex, stories: list[Story]) -> None:
    await index.add_all(
        [
            Document(
                id=story.id,
                text=story.text,
                meta={"genre": story.genre, "title": story.title, "words": story.word_count},
            )
            for story in stories
        ]
    )


# ------------------------------------------------------------------ the fixture


def test_fixture_records_its_embedder(embedder: ReplayEmbedder) -> None:
    """SPEC.md §2.1 applied to the recording itself.

    A bag of vectors with no model identity is exactly what makes a silent model
    swap possible, so the fixture carries one and the index inherits it.
    """
    assert embedder.id == "gemini-embedding-001@768"
    assert embedder.dimension == 768
    assert len(embedder) == 35  # 10 documents + 25 queries


def test_recorded_vectors_are_not_unit_norm(embedder: ReplayEmbedder) -> None:
    """The reason this fixture is worth having.

    ``gemini-embedding-001`` pre-normalizes only its default 3072-dimension
    output; at 768 the norms land near 0.59. The embedder reports that honestly
    rather than claiming a normalization it does not perform.
    """
    assert embedder.produces_normalized is False

    recorded = json.loads(FIXTURE.read_text(encoding="utf-8"))
    norms = [
        float(np.linalg.norm(np.asarray(record["vector"], dtype=np.float64)))
        for record in recorded["documents"]
    ]
    assert all(0.5 < norm < 0.7 for norm in norms), f"norms moved: {min(norms)}..{max(norms)}"
    assert recorded["normalized"] is False


async def test_write_time_normalization_fixes_unnormalized_input(story_index, corpus) -> None:
    """SPEC.md §3.1, exercised against real unnormalized vectors.

    Every other normalization test starts from a vector that was already
    unit-norm, so it could pass with the normalization step deleted. This one
    cannot.
    """
    index, stories = story_index
    await _fill(index, stories)

    vectors = np.memmap(index.path / fmt.VECTORS, dtype="<f4", mode="r", shape=(len(stories), 768))
    for row in range(len(stories)):
        norm = float(np.linalg.norm(np.asarray(vectors[row], dtype=np.float64)))
        assert abs(norm - 1.0) < 1e-5, f"row {row} has norm {norm}"


async def test_a_missing_recording_is_loud(story_index) -> None:
    """A mock that invents an answer is worse than one that fails."""
    index, _ = story_index
    with pytest.raises(ReplayMissError) as caught:
        await index.add_all([Document(id="x", text="text that was never embedded")])
    assert "never embedded" in str(caught.value)


async def test_document_and_query_recordings_are_separate(embedder: ReplayEmbedder, corpus) -> None:
    """The same text has two different correct vectors, by task type.

    Asking for a document embedding of a query string must miss rather than
    quietly serve the query vector — that substitution is a silent retrieval
    quality loss, and it is the reason the interface has two methods.
    """
    _, queries = corpus
    prompt = queries[0].prompt
    assert (await embedder.embed_query(prompt)).shape == (768,)
    with pytest.raises(ReplayMissError) as caught:
        await embedder.embed_documents([prompt])
    assert "recorded as a query" in str(caught.value)


# ------------------------------------------------------------ retrieval quality


async def test_top_1_accuracy_on_targeted_queries(story_index, corpus) -> None:
    """Measured at 18/20 when the fixture was recorded. Guarded at 17."""
    index, stories = story_index
    _, queries = corpus
    await _fill(index, stories)

    targeted = [query for query in queries if not query.is_negative]
    hits = 0
    misses: list[str] = []
    for query in targeted:
        results = await index.search(query.prompt, k=1)
        assert results, f"no result for {query.prompt!r}"
        if results[0].id == query.target:
            hits += 1
        else:
            misses.append(f"{query.prompt!r} -> {results[0].id} (wanted {query.target})")

    assert hits >= 17, f"top-1 accuracy fell to {hits}/{len(targeted)}:\n  " + "\n  ".join(misses)


async def test_the_target_is_always_in_the_top_three(story_index, corpus) -> None:
    index, stories = story_index
    _, queries = corpus
    await _fill(index, stories)

    for query in (q for q in queries if not q.is_negative):
        top3 = [result.id for result in await index.search(query.prompt, k=3)]
        assert query.target in top3, (
            f"{query.prompt!r} put {query.target} outside the top 3: {top3}"
        )


async def test_negative_controls_score_below_every_real_match(story_index, corpus) -> None:
    """The result worth having from this corpus.

    Five queries about tax returns, sourdough, quantum computing and bicycle
    brakes have no right answer. Brute force still returns k results for them —
    exact search cannot abstain — but their best score sits below the *worst*
    score of any genuine match. Measured: negatives peak at 0.5507, targeted
    queries bottom out at 0.5538.
    """
    index, stories = story_index
    _, queries = corpus
    await _fill(index, stories)

    async def best(query: Query) -> float:
        results = await index.search(query.prompt, k=1)
        return results[0].score

    targeted = [await best(q) for q in queries if not q.is_negative]
    negative = [await best(q) for q in queries if q.is_negative]

    assert negative, "the corpus lost its negative controls"
    assert max(negative) < min(targeted), (
        f"a negative control scored {max(negative):.4f}, at or above the weakest "
        f"real match at {min(targeted):.4f}"
    )


async def test_scores_cluster_in_a_narrow_band(story_index, corpus) -> None:
    """Why the API refuses to expose an absolute score threshold. SPEC.md §8.

    Across 250 query-document pairs every cosine similarity falls between about
    0.45 and 0.75. The signal is real but it is a 0.3-wide band sitting far from
    zero, and where the band sits moves with the model and the corpus. A
    ``score > 0.7`` rule would be tuned to this fixture and meaningless anywhere
    else — only the ordering carries information.
    """
    index, stories = story_index
    _, queries = corpus
    await _fill(index, stories)

    scores = [
        result.score
        for query in queries
        for result in await index.search(query.prompt, k=len(stories))
    ]
    assert len(scores) == len(queries) * len(stories)
    assert min(scores) > 0.2, "the band moved; the threshold argument needs remeasuring"
    assert max(scores) < 0.95
    assert max(scores) - min(scores) < 0.5, "scores span less than half the available range"


# ------------------------------------------------------------------ index behaviour


async def test_genre_filter_is_exact(story_index, corpus) -> None:
    index, stories = story_index
    _, queries = corpus
    await _fill(index, stories)

    gothic = [story for story in stories if "Gothic" in story.genre]
    assert gothic, "the corpus lost its Gothic Horror entry"

    results = await index.search(
        queries[0].prompt, k=10, filter=lambda meta: "Gothic" in str(meta.get("genre", ""))
    )
    assert [result.id for result in results] == [story.id for story in gothic]


async def test_reindexing_the_corpus_embeds_nothing(story_index, corpus) -> None:
    """The incremental hash, measured on a corpus where re-embedding costs money."""
    index, stories = story_index
    await _fill(index, stories)

    second = await index.add_all(
        [Document(id=story.id, text=story.text, meta={"genre": story.genre}) for story in stories]
    )
    assert second.skipped == len(stories)
    assert second.added == 0 and second.replaced == 0
    assert index.size() == len(stories)


async def test_deleting_the_top_hit_promotes_the_runner_up(story_index, corpus) -> None:
    index, stories = story_index
    _, queries = corpus
    await _fill(index, stories)

    query = next(q for q in queries if not q.is_negative)
    before = [result.id for result in await index.search(query.prompt, k=3)]
    index.delete(before[0])

    after = [result.id for result in await index.search(query.prompt, k=3)]
    assert before[0] not in after
    assert after[0] == before[1]


async def test_opening_the_story_index_with_the_hashing_embedder_is_refused(
    story_index, corpus
) -> None:
    """The guard, on an index where the mistake would be expensive.

    A 768-dimension Gemini index and a 64-dimension hashing index are not
    interchangeable, and the error says so by name rather than returning
    nonsense rankings.
    """
    index, stories = story_index
    await _fill(index, stories)
    path = index.path
    index.close()

    with pytest.raises((EmbedderMismatchError, CorruptIndexError)) as caught:
        SemanticIndex.open(path, HashingEmbedder(dimension=64))
    assert "gemini-embedding-001@768" in str(caught.value)


async def test_the_story_index_survives_a_reopen(tmp_path, corpus) -> None:
    stories, queries = corpus
    path = tmp_path / "stories.index"
    query = next(q for q in queries if not q.is_negative)

    with SemanticIndex.create(path, ReplayEmbedder.from_file(FIXTURE)) as index:
        await _fill(index, stories)
        expected = [(r.id, r.score) for r in await index.search(query.prompt, k=5)]

    with SemanticIndex.open(path, ReplayEmbedder.from_file(FIXTURE)) as reopened:
        assert reopened.live_count() == len(stories)
        assert [(r.id, r.score) for r in await reopened.search(query.prompt, k=5)] == expected
