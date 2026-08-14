package dev.simplesemantic

import io.kotest.core.spec.style.StringSpec
import io.kotest.engine.spec.tempdir
import io.kotest.matchers.collections.shouldContainExactly
import io.kotest.matchers.doubles.shouldBeGreaterThan
import io.kotest.matchers.doubles.shouldBeLessThan
import io.kotest.matchers.shouldBe
import kotlin.math.abs

/** Search behaviour. SPEC.md §8. */
class SearchSpec : StringSpec({

    fun freshIndex(): SemanticIndex =
        SemanticIndex.create(tempdir().toPath().resolve("index"), HashingEmbedder(dimension = 64))

    "an exact match ranks first" {
        freshIndex().use { index ->
            index.addAll(
                listOf(
                    Document("cats", "cats are small carnivorous mammals"),
                    Document("ships", "container ships move freight across oceans"),
                    Document("bread", "sourdough bread needs a starter culture"),
                ),
            )
            val results = index.search("cats are small carnivorous mammals", k = 3)
            results[0].id shouldBe "cats"
            results[0].score shouldBeGreaterThan results[1].score
        }
    }

    "scores are cosine similarities in range" {
        freshIndex().use { index ->
            index.addAll((0 until 5).map { Document("d$it", "topic $it words here") })
            for (result in index.search("topic words", k = 5)) {
                (result.score <= 1.0 + 1e-9) shouldBe true
                (result.score >= -1.0 - 1e-9) shouldBe true
            }
        }
    }

    "metadata round-trips into results" {
        // Stored and returned; the index does not filter on it.
        freshIndex().use { index ->
            index.addAll(listOf(Document("a", "shared subject matter", mapOf("source" to "a"))))
            index.search("shared subject matter", k = 1).single().meta shouldBe
                mapOf("source" to "a")
        }
    }

    "recall is total by construction" {
        // Every live row is scored, so the top-1 is the definitional top-1.
        freshIndex().use { index ->
            index.addAll((0 until 50).map { Document("d$it", "unique phrasing number $it zulu") })
            for (target in listOf(0, 17, 49)) {
                index.search("unique phrasing number $target zulu", k = 1)[0].id shouldBe "d$target"
            }
            index.search("unique phrasing", k = 50).size shouldBe 50
        }
    }

    "ties break by ascending row" {
        // Without this, the two implementations are comparing noise.
        freshIndex().use { index ->
            index.addAll((0 until 4).map { Document("same$it", "identical text") })
            val results = index.search("identical text", k = 4)
            results.map { it.row } shouldContainExactly listOf(0, 1, 2, 3)
            results.map { it.score }.toSet().size shouldBe 1
        }
    }

    "empty and whitespace queries return nothing" {
        freshIndex().use { index ->
            index.addAll(listOf(Document("a", "hello world")))
            index.search("", k = 5) shouldBe emptyList()
            index.search("   \n\t ", k = 5) shouldBe emptyList()
        }
    }

    "search on an empty index returns nothing" {
        freshIndex().use { index ->
            index.search("anything", k = 5) shouldBe emptyList()
            index.size() shouldBe 0
            index.liveCount() shouldBe 0
        }
    }

    "searchVector accepts a pre-computed query and normalizes it" {
        freshIndex().use { index ->
            index.addAll(listOf(Document("a", "alpha beta gamma")))
            val vector = HashingEmbedder(dimension = 64).embedText("alpha beta gamma")
            index.searchVector(vector, k = 1)[0].id shouldBe "a"

            val scaled = FloatArray(vector.size) { vector[it] * 17.0f }
            abs(index.searchVector(scaled, k = 1)[0].score - 1.0) shouldBeLessThan 1e-6
        }
    }

    "non-ascii text round-trips" {
        freshIndex().use { index ->
            val documents = listOf(
                Document("ru", "Привет мир, это тестовый документ"),
                Document("fr", "Café crème à la française, déjà vu"),
                Document("jp", "日本語のテキストを検索する"),
                Document("emoji", "rocket 🚀 launch sequence"),
            )
            index.addAll(documents)
            for (document in documents) {
                val results = index.search(document.text, k = 1)
                results[0].id shouldBe document.id
                results[0].text shouldBe document.text
            }
            index.get("jp")!!.text shouldBe "日本語のテキストを検索する"
        }
    }

    "the tokenizer keeps non-latin scripts" {
        // An ASCII-only regex returns an empty token list for all of these.
        tokenize("Привет мир") shouldContainExactly listOf("привет", "мир")
        tokenize("Café Crème") shouldContainExactly listOf("café", "crème")
        tokenize("日本語 テキスト") shouldContainExactly listOf("日本語", "テキスト")
        tokenize("snake_case and-dashes 42") shouldContainExactly
            listOf("snake", "case", "and", "dashes", "42")
    }

    "the tokenizer lowercases locale-independently" {
        // Locale.ROOT, not the default: a Turkish locale maps I to ı. İ
        // lowercases to i plus a combining dot, category Mn, which ends the
        // token. SPEC.md appendix A.
        tokenize("STRASSE Iİ") shouldContainExactly listOf("strasse", "ii")
        tokenize("ΟΔΟΣ") shouldContainExactly listOf("οδος")
    }

    "chunked documents are retrievable" {
        freshIndex().use { index ->
            val chunker = FixedChunker(size = 40, overlap = 8)
            val longText = (0 until 20).joinToString(" ") { "segment$it filler words here" }
            val pieces = chunker.chunk(longText)
            (pieces.size > 1) shouldBe true

            index.addAll(
                pieces.map { Document("long#${it.index}", it.text, mapOf("source" to "long")) },
            )
            index.search(pieces[3].text, k = 1)[0].id shouldBe "long#3"
        }
    }

    "chunk boundaries are code points" {
        // UTF-16 code units would split a surrogate pair.
        val pieces = FixedChunker(size = 4, overlap = 0).chunk("🚀".repeat(10))
        pieces.map { it.text } shouldContainExactly listOf("🚀🚀🚀🚀", "🚀🚀🚀🚀", "🚀🚀")
    }

    "the chunker rejects empty and whitespace text" {
        val chunker = FixedChunker(size = 16, overlap = 4)
        chunker.chunk("") shouldBe emptyList()
        chunker.chunk("   \n  ") shouldBe emptyList()
    }

    "ties at the k boundary keep the lowest rows" {
        // Selection must not drop a tied row for an equal-scoring later one:
        // a bare partition keeps an arbitrary subset of the tied rows.
        freshIndex().use { index ->
            index.addAll((0 until 8).map { Document("tie$it", "identical scoring text") })
            index.addAll(
                listOf(Document("best", "identical scoring text plus a distinguishing tail")),
            )

            val results = index.search("identical scoring text", k = 4)
            val tied = results.filter { it.id.startsWith("tie") }
            tied.map { it.row } shouldContainExactly tied.map { it.row }.sorted()
            tied.map { it.id } shouldContainExactly tied.indices.map { "tie$it" }
        }
    }

    "k boundary selection is stable across calls" {
        freshIndex().use { index ->
            index.addAll((0 until 12).map { Document("tie$it", "same text everywhere") })
            val first = index.search("same text everywhere", k = 5).map { it.id }
            repeat(5) {
                index.search("same text everywhere", k = 5).map { it.id } shouldBe first
            }
        }
    }
})
