package dev.simplesemantic

import io.kotest.core.spec.style.StringSpec
import io.kotest.engine.spec.tempdir
import io.kotest.matchers.collections.shouldContainExactly
import io.kotest.matchers.shouldBe
import io.kotest.matchers.string.shouldContain
import io.kotest.matchers.string.shouldNotContain
import java.nio.file.Files

/**
 * Update semantics. SPEC.md §6.
 *
 * Several of these are written specifically to catch the class of defect found
 * in the sibling project `simple-fts`, where an update left stale copies of a
 * document alive. The append-only row layout makes that shape of bug
 * structurally impossible here — these prove the claim rather than assuming it.
 */
class UpdateSpec : StringSpec({

    fun freshIndex(): SemanticIndex =
        SemanticIndex.create(tempdir().toPath().resolve("index"), HashingEmbedder(dimension = 64))

    "upsert returns only the new version" {
        // The `simple-fts` bug: a term shared by both versions returned the stale copy.
        freshIndex().use { index ->
            val shared = "shared vocabulary appears in both revisions"
            index.addAll(listOf(Document("doc", "$shared original wording")))
            index.upsert(Document("doc", "$shared revised wording"))

            val results = index.search(shared, k = 10)
            results.size shouldBe 1
            results[0].text shouldContain "revised"
            results[0].text shouldNotContain "original"
        }
    }

    "result ids are distinct across a query matching many rows" {
        freshIndex().use { index ->
            for (revision in 0 until 5) {
                index.upsert(Document("doc", "common terms revision $revision"))
            }
            index.addAll((0 until 3).map { Document("other$it", "common terms") })

            val ids = index.search("common terms", k = 20).map { it.id }
            ids.size shouldBe ids.toSet().size
        }
    }

    "live count after N upserts of one id is one" {
        freshIndex().use { index ->
            for (revision in 0 until 10) index.upsert(Document("doc", "revision $revision"))
            index.liveCount() shouldBe 1
            // Append-only: every revision is still a row.
            index.size() shouldBe 10
        }
    }

    "a tombstoned row never appears, even as the nearest vector" {
        freshIndex().use { index ->
            val query = "precisely this exact phrase"
            index.addAll(
                listOf(
                    Document("target", query),
                    Document("other", "a completely unrelated sentence about ferrets"),
                ),
            )
            index.search(query, k = 5)[0].id shouldBe "target"

            index.delete("target") shouldBe true
            index.search(query, k = 5).map { it.id } shouldContainExactly listOf("other")
        }
    }

    "deleting an id that was never added is a no-op" {
        freshIndex().use { index ->
            index.addAll(listOf(Document("a", "hello")))
            index.delete("never-added") shouldBe false
            index.liveCount() shouldBe 1
        }
    }

    "re-adding an identical document is a no-op" {
        // Driven by the content hash. This is the API cost control. SPEC.md §4.1.
        freshIndex().use { index ->
            val document = Document("a", "unchanged text", mapOf("k" to "v"))
            val first = index.addAll(listOf(document))
            val second = index.addAll(listOf(document))

            first.added shouldBe 1
            first.skipped shouldBe 0
            second.added shouldBe 0
            second.replaced shouldBe 0
            second.skipped shouldBe 1
            index.size() shouldBe 1
        }
    }

    "a duplicate id within one batch resolves to the last" {
        freshIndex().use { index ->
            index.addAll(
                listOf(
                    Document("a", "first version"),
                    Document("a", "second version"),
                ),
            )
            index.liveCount() shouldBe 1
            index.get("a")!!.text shouldBe "second version"
        }
    }

    "an existing row is never mutated in place" {
        freshIndex().use { index ->
            index.addAll(listOf(Document("a", "original")))
            val originalRow = VectorStore.open(index.path.resolve(FileNames.VECTORS), 1, 64)
                .use { it.row(0).toList() }

            index.upsert(Document("a", "replacement"))
            VectorStore.open(index.path.resolve(FileNames.VECTORS), 2, 64).use { store ->
                store.row(0).toList() shouldContainExactly originalRow
            }
        }
    }

    "compaction drops tombstones and preserves search results" {
        freshIndex().use { index ->
            index.addAll((0 until 6).map { Document("d$it", "document about topic $it") })
            index.delete("d1")
            index.delete("d4")
            index.upsert(Document("d0", "document about topic zero, revised"))

            val before = index.search("document about topic", k = 10).map { it.id to it.score }
            index.size() shouldBe 7
            index.liveCount() shouldBe 4

            index.compact() shouldBe 3
            index.size() shouldBe 4
            index.liveCount() shouldBe 4
            Files.exists(index.path.resolve(FileNames.TOMBSTONES)) shouldBe false

            index.search("document about topic", k = 10).map { it.id to it.score } shouldBe before
        }
    }

    "compaction is a no-op when nothing is deleted" {
        freshIndex().use { index ->
            index.addAll(listOf(Document("a", "hello")))
            index.compact() shouldBe 0
            index.size() shouldBe 1
        }
    }

    "a leftover staging directory does not block compaction" {
        freshIndex().use { index ->
            index.addAll((0 until 3).map { Document("d$it", "t$it") })
            index.delete("d1")

            val staging = index.path.resolveSibling("${index.path.fileName}.compact.tmp")
            Files.createDirectories(staging)
            Files.writeString(staging.resolve("leftover"), "debris from an interrupted run")

            index.compact() shouldBe 1
            index.liveCount() shouldBe 2
            Files.exists(staging) shouldBe false
        }
    }

    "state survives close and reopen" {
        val directory = tempdir().toPath().resolve("index")
        val expected = SemanticIndex.create(directory, HashingEmbedder(dimension = 64)).use { index ->
            index.addAll(
                (0 until 4).map { Document("d$it", "text $it", mapOf("n" to it.toLong())) },
            )
            index.delete("d2")
            index.upsert(Document("d0", "text zero revised"))
            index.search("text", k = 10).map { it.id to it.score }
        }

        SemanticIndex.open(directory, HashingEmbedder(dimension = 64)).use { reopened ->
            reopened.liveCount() shouldBe 3
            reopened.contains("d0") shouldBe true
            reopened.contains("d2") shouldBe false
            reopened.get("d0")!!.text shouldBe "text zero revised"
            reopened.search("text", k = 10).map { it.id to it.score } shouldBe expected
        }
    }

    "k larger than the live count returns all live rows" {
        freshIndex().use { index ->
            index.addAll((0 until 3).map { Document("d$it", "document $it") })
            for (k in listOf(0, 1, 3, 50)) {
                index.search("document", k = k).size shouldBe minOf(k, 3)
            }
        }
    }
})
