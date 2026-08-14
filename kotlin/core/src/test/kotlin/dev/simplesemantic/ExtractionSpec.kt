package dev.simplesemantic

import io.kotest.assertions.throwables.shouldThrow
import io.kotest.core.spec.style.StringSpec
import io.kotest.engine.spec.tempdir
import io.kotest.matchers.shouldBe
import io.kotest.matchers.string.shouldContain

/**
 * Document extraction. Both entry points are public — see Extraction.kt.
 *
 * This is the README example verbatim, which is the point: `simple-fts`
 * shipped an extractor that crashed on exactly the nullable field its own
 * README declared, because the example was never a test.
 */
data class Note(
    @SemanticId val id: String,
    @SemanticIndexed(order = 0) val title: String,
    @SemanticIndexed(order = 1) val body: String?,
    @SemanticMeta val tags: List<String> = emptyList(),
)

class ExtractionSpec : StringSpec({

    "annotated extraction" {
        val documents = documentsFrom(
            listOf(Note("n1", "Embeddings", "Dense vectors.", listOf("ml"))),
        )
        documents.size shouldBe 1
        documents[0].id shouldBe "n1"
        documents[0].text shouldBe "Embeddings\n\nDense vectors."
        documents[0].meta shouldBe mapOf("tags" to listOf("ml"))
    }

    "a nullable indexed field does not throw" {
        val documents = documentsFrom(listOf(Note("n1", "Title only", null)))
        documents[0].text shouldBe "Title only"
    }

    "order pins the concatenation" {
        // getDeclaredFields() is not specified to return declaration order, so
        // the annotation carries it explicitly.
        documentsFrom(listOf(Note("n1", "A", "B"))) [0].text shouldBe "A\n\nB"
    }

    "a missing id annotation names the annotation and the class" {
        val error = shouldThrow<ExtractionException> { documentsFrom(listOf(NoId("x"))) }
        error.message!! shouldContain "NoId"
        error.message!! shouldContain "@SemanticId"
    }

    "a missing indexed annotation is an error" {
        val error = shouldThrow<ExtractionException> { documentsFrom(listOf(NoText("x"))) }
        error.message!! shouldContain "@SemanticIndexed"
    }

    "two id fields is an error" {
        val error = shouldThrow<ExtractionException> { documentsFrom(listOf(TwoIds("1", "2", "t"))) }
        error.message!! shouldContain "exactly one"
    }

    "a null id is an error naming the field" {
        val error = shouldThrow<ExtractionException> { documentsFrom(listOf(NullableId(null, "t"))) }
        error.message!! shouldContain "NullableId.id"
    }

    "the lambda factory is public and needs no annotations" {
        val rows = listOf(mapOf("key" to "r1", "body" to "some text", "source" to "db"))
        val documents = documentsFrom(
            rows,
            idOf = { it.getValue("key") },
            textOf = { it.getValue("body") },
            metaOf = { mapOf("source" to it.getValue("source")) },
        )
        documents[0].id shouldBe "r1"
        documents[0].text shouldBe "some text"
        documents[0].meta shouldBe mapOf("source" to "db")
    }

    "the lambda factory's meta is optional" {
        val documents = documentsFrom(
            listOf("a" to "text"),
            idOf = { it.first },
            textOf = { it.second },
        )
        documents[0].meta shouldBe emptyMap()
    }

    "empty input yields no documents" {
        documentsFrom(emptyList<Note>()) shouldBe emptyList()
        documentsFrom(emptyList<String>(), idOf = { it }, textOf = { it }) shouldBe emptyList()
    }

    "extracted documents index and search" {
        val index = SemanticIndex.create(
            tempdir().toPath().resolve("index"),
            HashingEmbedder(dimension = 64),
        )
        index.use {
            it.addAll(
                documentsFrom(
                    listOf(
                        Note("n1", "Vector search", "Cosine similarity over a matrix.", listOf("ir")),
                        Note("n2", "Bread", null, listOf("food")),
                    ),
                ),
            )
            val results = it.search("Cosine similarity over a matrix.", k = 1)
            results[0].id shouldBe "n1"
            results[0].meta shouldBe mapOf("tags" to listOf("ir"))
        }
    }
})

private data class NoId(@SemanticIndexed val title: String)

private data class NoText(@SemanticId val id: String)

private data class TwoIds(
    @SemanticId val a: String,
    @SemanticId val b: String,
    @SemanticIndexed val t: String,
)

private data class NullableId(
    @SemanticId val id: String?,
    @SemanticIndexed val t: String,
)
