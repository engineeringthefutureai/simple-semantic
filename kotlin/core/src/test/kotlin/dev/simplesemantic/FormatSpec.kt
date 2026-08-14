package dev.simplesemantic

import io.kotest.assertions.throwables.shouldThrow
import io.kotest.core.spec.style.StringSpec
import io.kotest.engine.spec.tempdir
import io.kotest.matchers.collections.shouldContainExactly
import io.kotest.matchers.doubles.shouldBeLessThan
import io.kotest.matchers.shouldBe
import io.kotest.matchers.shouldNotBe
import io.kotest.matchers.string.shouldContain
import io.kotest.matchers.string.shouldNotContain
import java.nio.file.Files
import kotlin.math.abs

/**
 * The on-disk format. SPEC.md §2-§7.
 *
 * Each test builds its own index. Sharing mutable index state across cases —
 * which the `simple-fts` specs did — makes every failure ambiguous: a broken
 * delete shows up as a failure in an unrelated search case three tests later.
 */
class FormatSpec : StringSpec({

    fun freshIndex(name: String = "index"): SemanticIndex =
        SemanticIndex.create(tempdir().toPath().resolve(name), HashingEmbedder(dimension = 64))

    "files have the shapes the manifest declares" {
        freshIndex().use { index ->
            index.addAll((0 until 5).map { Document("d$it", "document number $it") })

            val manifest = index.manifest
            manifest.rowCount shouldBe 5
            manifest.liveCount shouldBe 5
            Files.size(index.path.resolve(FileNames.VECTORS)) shouldBe 5L * 64 * 4
            Files.size(index.path.resolve(FileNames.OFFSETS)) shouldBe 6L * 8
            // No deletions, so the tombstone file need not exist at all.
            Files.exists(index.path.resolve(FileNames.TOMBSTONES)) shouldBe false
        }
    }

    "docs.jsonl line N describes vector row N" {
        freshIndex().use { index ->
            val texts = listOf("alpha beta", "gamma delta", "epsilon zeta")
            index.addAll(texts.mapIndexed { i, text -> Document("d$i", text) })

            val lines = Files.readAllLines(index.path.resolve(FileNames.DOCS))
            VectorStore.open(index.path.resolve(FileNames.VECTORS), 3, 64).use { store ->
                for (row in texts.indices) {
                    CanonicalJson.parseObject(lines[row])["text"] shouldBe texts[row]
                    val expected = HashingEmbedder(dimension = 64).embedText(texts[row])
                    store.row(row).toList() shouldContainExactly expected.toList()
                }
            }
        }
    }

    "every stored row is unit norm" {
        freshIndex().use { index ->
            index.addAll(
                listOf(
                    Document("a", "the quick brown fox"),
                    Document("b", "jumps over the lazy dog"),
                    Document("c", "x"),
                ),
            )
            VectorStore.open(index.path.resolve(FileNames.VECTORS), 3, 64).use { store ->
                for (row in 0 until 3) {
                    var sum = 0.0
                    for (value in store.row(row)) sum += value.toDouble() * value
                    abs(Math.sqrt(sum) - 1.0) shouldBeLessThan 1e-5
                }
            }
        }
    }

    "offsets index exactly the lines" {
        freshIndex().use { index ->
            index.addAll((0 until 4).map { Document("d$it", "text $it") })
            val raw = Files.readAllBytes(index.path.resolve(FileNames.DOCS))
            val offsets = Files.readAllBytes(index.path.resolve(FileNames.OFFSETS))

            offsets.size shouldBe 5 * 8
            fun offsetAt(i: Int): Int {
                var value = 0L
                for (b in 7 downTo 0) value = (value shl 8) or (offsets[i * 8 + b].toLong() and 0xFF)
                return value.toInt()
            }
            offsetAt(4) shouldBe raw.size
            for (row in 0 until 4) {
                val line = String(raw, offsetAt(row), offsetAt(row + 1) - offsetAt(row))
                line.endsWith("\n") shouldBe true
                CanonicalJson.parseObject(line.trimEnd('\n'))["id"] shouldBe "d$row"
            }
        }
    }

    "vectors are written little-endian" {
        // SPEC.md §3. The JVM defaults to big-endian, so this must be pinned.
        // Asserted at the byte level rather than by round-trip, because a
        // round-trip through one implementation passes with either convention
        // — it is only the *other* implementation that notices.
        val bytes = floatArrayOf(1.0f, -2.0f, 0.5f).toLittleEndianBytes()
        // 1.0f is 0x3F800000; little-endian on disk is 00 00 80 3F.
        bytes.copyOfRange(0, 4).toList() shouldContainExactly
            listOf(0x00.toByte(), 0x00.toByte(), 0x80.toByte(), 0x3F.toByte())
    }

    "tombstone bits are LSB-first" {
        // MSB-first parses without error and means something else entirely.
        val bits = Tombstones.empty(10)
        bits.markDeleted(0)
        bits.markDeleted(9)
        val raw = bits.toByteArray()
        raw.size shouldBe 2
        raw[0] shouldBe 0b0000_0001.toByte()
        raw[1] shouldBe 0b0000_0010.toByte()
        bits.liveMask().toList() shouldContainExactly
            listOf(false) + List(8) { true } + listOf(false)
    }

    "content hash separates its three inputs" {
        contentHash("abc", "e", "c") shouldNotBe contentHash("ab", "ce", "c")
        contentHash("abc", "e", "c") shouldBe contentHash("abc", "e", "c")
        // Changing the embedder invalidates the vector even when the text is equal.
        contentHash("abc", "e1", "c") shouldNotBe contentHash("abc", "e2", "c")
    }

    "canonical json has no whitespace and sorts meta keys" {
        CanonicalJson.encodeObject(linkedMapOf("b" to 1, "a" to 2), sortKeys = true) shouldBe
            """{"a":2,"b":1}"""
        CanonicalJson.encodeDocument("id", "text", linkedMapOf("z" to 1, "a" to "x"), "hash") shouldBe
            """{"id":"id","text":"text","meta":{"a":"x","z":1},"hash":"hash"}"""
    }

    "canonical json emits non-ascii literally" {
        val line = CanonicalJson.encodeDocument("id", "Привет 日本語 café", emptyMap(), "h")
        line shouldContain "Привет 日本語 café"
        line shouldNotContain "\\u"
    }

    "canonical json escapes control characters with lowercase hex" {
        CanonicalJson.encodeString("a\u001Fb\n") shouldBe "\"a\\u001fb\\n\""
    }

    "meta rejects floating-point values" {
        // SPEC.md §7.2 — not because floats are hard, but because their
        // decimal form is not portable across languages.
        val error = shouldThrow<MetaValueException> {
            CanonicalJson.encodeDocument("id", "text", mapOf("score" to 0.5), "h")
        }
        error.message!! shouldContain "score"
    }

    "meta is validated before the embedder is called" {
        var calls = 0
        val counting = object : Embedder by HashingEmbedder(dimension = 64) {
            override suspend fun embedDocuments(texts: List<String>): List<FloatArray> {
                calls++
                return HashingEmbedder(dimension = 64).embedDocuments(texts)
            }
        }
        SemanticIndex.create(tempdir().toPath().resolve("i"), counting).use { index ->
            shouldThrow<MetaValueException> {
                index.addAll(listOf(Document("a", "hello", mapOf("score" to 0.5))))
            }
        }
        calls shouldBe 0
    }

    "open refuses a different embedder" {
        // SPEC.md §2.1: the single most important correctness rule here.
        val directory = tempdir().toPath().resolve("index")
        SemanticIndex.create(directory, HashingEmbedder(dimension = 64)).use { index ->
            index.addAll(listOf(Document("a", "hello")))
        }
        val error = shouldThrow<EmbedderMismatchException> {
            SemanticIndex.open(directory, HashingEmbedder(dimension = 64, seed = 1))
        }
        error.message!! shouldContain "hashing-0@64"
        error.message!! shouldContain "hashing-1@64"
    }

    "open refuses a truncated vectors file" {
        val directory = tempdir().toPath().resolve("index")
        SemanticIndex.create(directory, HashingEmbedder(dimension = 64)).use { index ->
            index.addAll((0 until 3).map { Document("d$it", "t$it") })
        }
        val file = directory.resolve(FileNames.VECTORS)
        val raw = Files.readAllBytes(file)
        Files.write(file, raw.copyOfRange(0, raw.size - 8))

        val error = shouldThrow<CorruptIndexException> {
            SemanticIndex.open(directory, HashingEmbedder(dimension = 64))
        }
        error.message!! shouldContain "truncated"
    }

    "open refuses an unknown format version" {
        val directory = tempdir().toPath().resolve("index")
        SemanticIndex.create(directory, HashingEmbedder(dimension = 64)).use { }
        val manifestFile = directory.resolve(FileNames.MANIFEST)
        Files.writeString(
            manifestFile,
            Files.readString(manifestFile).replace("\"format_version\":1", "\"format_version\":99"),
        )
        val error = shouldThrow<FormatVersionException> {
            SemanticIndex.open(directory, HashingEmbedder(dimension = 64))
        }
        error.message!! shouldContain "99"
    }

    "normalization handles the zero vector" {
        // 0/0 would put a NaN in the matrix that poisons every dot product.
        val out = normalizeRow(FloatArray(8))
        out.none { it.isNaN() } shouldBe true
        out[0] shouldBe 1.0f
    }

    "normalization is idempotent" {
        val once = normalizeRow(floatArrayOf(3.0f, 4.0f, 0.0f))
        normalizeRow(once).toList() shouldContainExactly once.toList()
    }

    "json parser round-trips what the encoder writes" {
        val meta = linkedMapOf<String, Any?>(
            "s" to "va\"lue\n",
            "n" to 42L,
            "b" to true,
            "nil" to null,
            "list" to listOf("a", 1L, false),
            "nested" to mapOf("k" to "v"),
        )
        val encoded = CanonicalJson.encodeDocument("id", "text", meta, "hash")
        val parsed = CanonicalJson.parseObject(encoded)
        parsed["id"] shouldBe "id"
        @Suppress("UNCHECKED_CAST")
        val roundTripped = parsed["meta"] as Map<String, Any?>
        roundTripped["s"] shouldBe "va\"lue\n"
        roundTripped["n"] shouldBe 42L
        roundTripped["list"] shouldBe listOf("a", 1L, false)
    }
})
