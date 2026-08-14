"""Search behaviour. SPEC.md §8."""

from __future__ import annotations

import numpy as np

from simple_semantic import Document, FixedChunker, SemanticIndex, tokenize


async def test_exact_match_ranks_first(index: SemanticIndex) -> None:
    await index.add_all(
        [
            Document(id="cats", text="cats are small carnivorous mammals"),
            Document(id="ships", text="container ships move freight across oceans"),
            Document(id="bread", text="sourdough bread needs a starter culture"),
        ]
    )
    results = await index.search("cats are small carnivorous mammals", k=3)
    assert results[0].id == "cats"
    assert results[0].score > results[1].score


async def test_scores_are_cosine_similarities_in_range(index: SemanticIndex) -> None:
    await index.add_all([Document(id=f"d{i}", text=f"topic {i} words here") for i in range(5)])
    for result in await index.search("topic words", k=5):
        assert -1.0 - 1e-9 <= result.score <= 1.0 + 1e-9


async def test_filter_is_exact_and_applied_before_scoring(index: SemanticIndex) -> None:
    await index.add_all(
        [
            Document(id="a1", text="shared subject matter", meta={"source": "a"}),
            Document(id="a2", text="shared subject matter", meta={"source": "a"}),
            Document(id="b1", text="shared subject matter", meta={"source": "b"}),
        ]
    )
    results = await index.search("shared subject matter", k=10, filter=lambda m: m["source"] == "b")
    assert [r.id for r in results] == ["b1"]


async def test_filter_matching_nothing_returns_nothing(index: SemanticIndex) -> None:
    await index.add_all([Document(id="a", text="hello", meta={"source": "a"})])
    assert await index.search("hello", k=5, filter=lambda m: m.get("source") == "zzz") == []


async def test_full_selectivity_filter_returns_every_live_row(index: SemanticIndex) -> None:
    """The exactness claim: a filter that admits everything changes nothing but the work done."""
    await index.add_all([Document(id=f"d{i}", text=f"row {i}", meta={"n": i}) for i in range(20)])
    unfiltered = await index.search("row", k=20)
    filtered = await index.search("row", k=20, filter=lambda m: True)
    assert [(r.id, r.score) for r in unfiltered] == [(r.id, r.score) for r in filtered]


async def test_recall_is_total_by_construction(index: SemanticIndex) -> None:
    """Every live row is scored, so the true nearest neighbour is always found.

    This is the property an ANN index trades away. Here it needs no tuning
    parameter and no recall measurement: k = live_count returns the entire
    corpus ranked, and the brute-force top-1 is the definitional top-1.
    """
    docs = [Document(id=f"d{i}", text=f"unique phrasing number {i} zulu") for i in range(50)]
    await index.add_all(docs)

    for target in (0, 17, 49):
        results = await index.search(f"unique phrasing number {target} zulu", k=1)
        assert results[0].id == f"d{target}"

    assert len(await index.search("unique phrasing", k=50)) == 50


async def test_ties_break_by_ascending_row(index: SemanticIndex) -> None:
    """SPEC.md §8. Without this, two implementations compare noise."""
    await index.add_all([Document(id=f"same{i}", text="identical text") for i in range(4)])
    results = await index.search("identical text", k=4)
    assert [r.row for r in results] == [0, 1, 2, 3]
    assert len({round(r.score, 12) for r in results}) == 1


async def test_empty_and_whitespace_queries_return_nothing(index: SemanticIndex) -> None:
    await index.add_all([Document(id="a", text="hello world")])
    assert await index.search("", k=5) == []
    assert await index.search("   \n\t ", k=5) == []


async def test_search_on_an_empty_index(index: SemanticIndex) -> None:
    assert await index.search("anything", k=5) == []
    assert index.size() == 0 and index.live_count() == 0


async def test_search_vector_accepts_a_precomputed_query(index: SemanticIndex) -> None:
    await index.add_all([Document(id="a", text="alpha beta gamma")])
    vector = index.embedder.embed_text("alpha beta gamma")
    results = index.search_vector(vector, k=1)
    assert results[0].id == "a"
    assert abs(results[0].score - 1.0) < 1e-6


async def test_search_vector_normalizes_an_unnormalized_query(index: SemanticIndex) -> None:
    await index.add_all([Document(id="a", text="alpha beta gamma")])
    vector = index.embedder.embed_text("alpha beta gamma") * np.float32(17.0)
    assert abs(index.search_vector(vector, k=1)[0].score - 1.0) < 1e-6


async def test_non_ascii_round_trips(index: SemanticIndex) -> None:
    """Cyrillic, accented Latin, CJK, and an astral-plane character."""
    docs = [
        Document(id="ru", text="Привет мир, это тестовый документ"),
        Document(id="fr", text="Café crème à la française, déjà vu"),
        Document(id="jp", text="日本語のテキストを検索する"),
        Document(id="emoji", text="rocket 🚀 launch sequence"),
    ]
    await index.add_all(docs)

    for doc in docs:
        results = await index.search(doc.text, k=1)
        assert results[0].id == doc.id, f"{doc.id} did not match itself"
        assert results[0].text == doc.text

    stored = index.get("jp")
    assert stored is not None and stored.text == "日本語のテキストを検索する"


def test_tokenizer_keeps_non_latin_scripts() -> None:
    """An ASCII-only regex would silently return an empty token list here."""
    assert tokenize("Привет мир") == ["привет", "мир"]
    assert tokenize("Café Crème") == ["café", "crème"]
    assert tokenize("日本語 テキスト") == ["日本語", "テキスト"]
    assert tokenize("snake_case and-dashes 42") == ["snake", "case", "and", "dashes", "42"]


def test_tokenizer_lowercases_locale_independently() -> None:
    """Locale-independent casing, and combining marks are not token characters.

    ``İ`` (U+0130) lowercases to ``i`` plus a combining dot above, and the
    combining mark is category Mn — outside ``\\p{L}\\p{N}`` — so it ends the
    token. The JVM's ``lowercase(Locale.ROOT)`` applies the same SpecialCasing
    rule and its ``\\p{L}\\p{N}`` excludes Mn identically, which is why
    conformance holds. A Turkish locale would map ``I`` to ``ı`` instead and
    the two implementations would diverge on the same input.
    """
    assert tokenize("STRASSE Iİ") == ["strasse", "ii"]
    assert tokenize("ΟΔΟΣ") == ["οδος"]


async def test_chunked_documents_are_retrievable(index: SemanticIndex) -> None:
    chunker = FixedChunker(size=40, overlap=8)
    long_text = " ".join(f"segment{i} filler words here" for i in range(20))
    pieces = chunker.chunk(long_text)
    assert len(pieces) > 1

    await index.add_all(
        [Document(id=f"long#{p.index}", text=p.text, meta={"source": "long"}) for p in pieces]
    )
    results = await index.search(pieces[3].text, k=1)
    assert results[0].id == "long#3"


def test_chunker_boundaries_are_code_points() -> None:
    """UTF-16 code units would split an emoji and make the JVM disagree."""
    text = "🚀" * 10
    pieces = FixedChunker(size=4, overlap=0).chunk(text)
    assert [p.text for p in pieces] == ["🚀🚀🚀🚀", "🚀🚀🚀🚀", "🚀🚀"]


def test_chunker_rejects_empty_and_whitespace_text() -> None:
    chunker = FixedChunker(size=16, overlap=4)
    assert chunker.chunk("") == []
    assert chunker.chunk("   \n  ") == []
