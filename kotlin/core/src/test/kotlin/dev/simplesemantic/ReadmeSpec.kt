package dev.simplesemantic

import io.kotest.core.spec.style.StringSpec
import io.kotest.engine.spec.tempdir
import io.kotest.matchers.collections.shouldContainExactly
import io.kotest.matchers.shouldBe

/**
 * The README examples, executed, so a stale snippet fails the build.
 *
 * Keep these in sync with README.md by hand; the only difference permitted is
 * the index path.
 */
class ReadmeSpec : StringSpec({

    "readme quickstart" {
        val embedder = HashingEmbedder(dimension = 256)
        SemanticIndex.create(tempdir().toPath().resolve("notes.index"), embedder).use { index ->
            index.addAll(
                listOf(
                    Document(
                        "n1",
                        "Cosine similarity over a dense matrix.",
                        mapOf("source" to "notes", "tags" to listOf("ir")),
                    ),
                    Document("n2", "Sourdough needs a mature starter."),
                ),
            )

            val hits = index.search("vector similarity", k = 5)
            hits.map { "%+.4f  %s".format(it.score, it.id) }.isNotEmpty() shouldBe true

            index.upsert(Document("n1", "Revised text."))
            index.delete("n2")
            index.compact()

            index.liveCount() shouldBe 1
            index.size() shouldBe 1
            index.get("n1")!!.text shouldBe "Revised text."
        }
    }

    "readme extraction" {
        val documents = documentsFrom(listOf(Note("n1", "Embeddings", null, listOf("ml"))))
        documents[0].id shouldBe "n1"
        // The nullable body is skipped rather than throwing.
        documents[0].text shouldBe "Embeddings"

        val rows = listOf(mapOf("key" to "r1", "body" to "some text"))
        documentsFrom(rows, idOf = { it.getValue("key") }, textOf = { it.getValue("body") })
            .first().id shouldBe "r1"
    }
})
