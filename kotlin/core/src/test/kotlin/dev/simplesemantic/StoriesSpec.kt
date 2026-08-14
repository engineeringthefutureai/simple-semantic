package dev.simplesemantic

import io.kotest.assertions.throwables.shouldThrow
import io.kotest.core.spec.style.StringSpec
import io.kotest.engine.spec.tempdir
import io.kotest.assertions.withClue
import io.kotest.matchers.collections.shouldContain
import io.kotest.matchers.doubles.shouldBeLessThan
import io.kotest.matchers.shouldBe
import io.kotest.matchers.string.shouldContain
import java.nio.file.Files
import java.nio.file.Path
import kotlin.math.abs

/**
 * Retrieval quality over the story corpus, with real embeddings.
 *
 * Ten stories and twenty-five queries whose vectors came from
 * `gemini-embedding-001`, replayed from a fixture. The same fixture drives the
 * Python suite; both must agree.
 */
class StoriesSpec : StringSpec({

    // Gradle runs tests with the subproject as the working directory, so walk
    // up until the conformance directory appears rather than counting parents.
    val repoRoot = generateSequence(Path.of("").toAbsolutePath()) { it.parent }
        .first { Files.isDirectory(it.resolve("conformance/fixtures")) }
    val fixture = repoRoot.resolve("conformance/fixtures/story-embeddings-v1.json")
    val storiesDir = repoRoot.resolve("conformance/stories")

    fun recorded(): FixtureWire =
        WireJson.decodeFromString(
            FixtureWire.serializer(),
            Files.readString(fixture, Charsets.UTF_8),
        )

    /** Story id to text. The fixture carries ids and filenames beside the vectors. */
    fun stories(): List<Pair<String, String>> =
        recorded().documents.map { it.id to Files.readString(storiesDir.resolve(it.file)) }

    /** Prompt to target-story id; a blank target marks a negative control. */
    fun queries(): List<Pair<String, String>> =
        recorded().queries.map { it.prompt to it.target }

    fun index(): SemanticIndex = SemanticIndex.create(
        tempdir().toPath().resolve("stories.index"),
        ReplayEmbedder.fromFile(fixture),
    )

    suspend fun SemanticIndex.fill(): List<Pair<String, String>> {
        val corpus = stories()
        addAll(corpus.map { (id, text) -> Document(id, text) })
        return corpus
    }

    "the fixture records its embedder" {
        // SPEC.md §2.1 applied to the recording itself.
        val embedder = ReplayEmbedder.fromFile(fixture)
        embedder.id shouldBe "gemini-embedding-001@768"
        embedder.dimension shouldBe 768
        embedder.size shouldBe 35 // 10 documents + 25 queries
    }

    "recorded vectors are not unit norm" {
        // At 768 dimensions the norms land near 0.59, and the embedder says so.
        val embedder = ReplayEmbedder.fromFile(fixture)
        embedder.producesNormalized shouldBe false
        val vector = embedder.embedDocuments(listOf(stories().first().second)).single()
        var sum = 0.0
        for (value in vector) sum += value.toDouble() * value
        val norm = Math.sqrt(sum)
        (norm > 0.5 && norm < 0.7) shouldBe true
    }

    "write-time normalization fixes unnormalized input" {
        // SPEC.md §3.1 against real unnormalized input: every other
        // normalization test starts from a unit-norm vector.
        index().use { index ->
            val corpus = index.fill()
            VectorStore.open(
                index.path.resolve(FileNames.VECTORS),
                corpus.size,
                768,
            ).use { store ->
                for (row in corpus.indices) {
                    var sum = 0.0
                    for (value in store.row(row)) sum += value.toDouble() * value
                    abs(Math.sqrt(sum) - 1.0) shouldBeLessThan 1e-5
                }
            }
        }
    }

    "a missing recording is loud" {
        // A mock that invents an answer is worse than one that fails.
        index().use { index ->
            val error = shouldThrow<ReplayMissException> {
                index.addAll(listOf(Document("x", "text that was never embedded")))
            }
            error.message!! shouldContain "never embedded"
        }
    }

    "document and query recordings are separate" {
        // The same text has two different correct vectors, by task type.
        val embedder = ReplayEmbedder.fromFile(fixture)
        val prompt = queries().first().first
        embedder.embedQuery(prompt).size shouldBe 768
        val error = shouldThrow<ReplayMissException> { embedder.embedDocuments(listOf(prompt)) }
        error.message!! shouldContain "recorded as a query"
    }

    "top-1 accuracy on targeted queries" {
        // Measured at 18/20 when the fixture was recorded. Guarded at 17.
        index().use { index ->
            index.fill()
            val targeted = queries().filter { it.second.isNotEmpty() }
            val misses = ArrayList<String>()
            var hits = 0
            for ((prompt, target) in targeted) {
                val results = index.search(prompt, k = 1)
                if (results.firstOrNull()?.id == target) {
                    hits++
                } else {
                    misses.add("'$prompt' -> ${results.firstOrNull()?.id} (wanted $target)")
                }
            }
            withClue("misses: $misses") { (hits >= 17) shouldBe true }
        }
    }

    "the target is always in the top three" {
        index().use { index ->
            index.fill()
            for ((prompt, target) in queries().filter { it.second.isNotEmpty() }) {
                index.search(prompt, k = 3).map { it.id } shouldContain target
            }
        }
    }

    "negative controls score below every real match" {
        // Exact search cannot abstain, so the five no-answer queries still
        // return k results — but below the weakest genuine match.
        index().use { index ->
            index.fill()
            val all = queries()
            val targeted = all.filter { it.second.isNotEmpty() }
                .map { index.search(it.first, k = 1).first().score }
            val negative = all.filter { it.second.isEmpty() }
                .map { index.search(it.first, k = 1).first().score }

            negative.isNotEmpty() shouldBe true
            (negative.max() < targeted.min()) shouldBe true
        }
    }

    "scores cluster in a narrow band" {
        // Why there is no absolute score threshold. SPEC.md §8. All 250 pairs
        // sit in a 0.3-wide band far from zero.
        index().use { index ->
            val corpus = index.fill()
            val scores = queries().flatMap { index.search(it.first, k = corpus.size) }.map { it.score }
            scores.size shouldBe queries().size * corpus.size
            (scores.min() > 0.2) shouldBe true
            (scores.max() < 0.95) shouldBe true
            (scores.max() - scores.min() < 0.5) shouldBe true
        }
    }

    "re-indexing the corpus embeds nothing" {
        index().use { index ->
            val corpus = index.fill()
            val second = index.addAll(corpus.map { (id, text) -> Document(id, text) })
            second.skipped shouldBe corpus.size
            second.added shouldBe 0
            second.replaced shouldBe 0
        }
    }

    "opening the story index with the hashing embedder is refused" {
        val directory = tempdir().toPath().resolve("stories.index")
        SemanticIndex.create(directory, ReplayEmbedder.fromFile(fixture)).use { it.fill() }

        val error = shouldThrow<EmbedderMismatchException> {
            SemanticIndex.open(directory, HashingEmbedder(dimension = 64))
        }
        error.message!! shouldContain "gemini-embedding-001@768"
    }
})
