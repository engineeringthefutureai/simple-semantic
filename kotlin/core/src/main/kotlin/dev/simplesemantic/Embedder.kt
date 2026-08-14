package dev.simplesemantic

import java.security.MessageDigest
import java.util.Locale

/**
 * The one boundary that matters.
 *
 * `embedDocuments` and `embedQuery` stay separate: Gemini needs
 * `taskType: RETRIEVAL_DOCUMENT` versus `RETRIEVAL_QUERY`, e5/BGE-style models
 * need `passage: ` / `query: ` prefixes. A single `embed()` makes that
 * asymmetry unrepresentable.
 *
 * Batching, retry and rate-limit handling belong to the implementation.
 */
public interface Embedder {
    /** Identity of the model, recorded in the manifest. SPEC.md §2.1. */
    public val id: String

    public val dimension: Int

    public val maxBatchSize: Int

    /** Honest report of whether the model's own output is already unit-norm. */
    public val producesNormalized: Boolean

    public suspend fun embedDocuments(texts: List<String>): List<FloatArray>

    public suspend fun embedQuery(text: String): FloatArray
}

/**
 * Lowercase, Unicode-aware tokenization. SPEC.md appendix A.
 *
 * `\p{L}\p{N}` rather than `[a-z0-9]`, which returns an empty token list for
 * every non-Latin script. `Locale.ROOT` rather than the default: under a
 * Turkish locale `I` lowercases to `ı`, so the same corpus would index
 * differently per machine.
 */
private val TOKEN_PATTERN = Regex("[\\p{L}\\p{N}]+")

public fun tokenize(text: String): List<String> =
    TOKEN_PATTERN.findAll(text.lowercase(Locale.ROOT)).map { it.value }.toList()

/**
 * Deterministic, seeded, no network. SPEC.md appendix A.
 *
 * A signed hashing trick, not a good embedding — it exists to be identical in
 * both languages bit for bit, which real models are not, and to make the test
 * suite runnable with no API key.
 *
 * Documents and queries share the embedding function here so that search over a
 * hashed index still returns sensible neighbours.
 */
public class HashingEmbedder(
    override val dimension: Int = 256,
    public val seed: Long = 0,
) : Embedder {

    init {
        require(dimension > 0) { "dimension must be positive, got $dimension" }
    }

    override val id: String = "hashing-$seed@$dimension"
    override val maxBatchSize: Int = 4096
    override val producesNormalized: Boolean = true

    private val seedBytes: ByteArray = ByteArray(8) { i -> ((seed ushr (i * 8)) and 0xFF).toByte() }

    /** Synchronous single-text embedding. No I/O, so nothing to suspend for. */
    public fun embedText(text: String): FloatArray {
        val accumulator = DoubleArray(dimension)
        val digest = MessageDigest.getInstance("SHA-256")
        for (token in tokenize(text)) {
            digest.reset()
            digest.update(token.toByteArray(Charsets.UTF_8))
            digest.update(0)
            digest.update(seedBytes)
            val hash = digest.digest()
            val column = (readUInt32LittleEndian(hash) % dimension.toLong()).toInt()
            val sign = if ((hash[4].toInt() and 1) == 0) 1.0 else -1.0
            accumulator[column] += sign
        }
        // Counts are small integers, exact in both Double and Float.
        val row = FloatArray(dimension) { accumulator[it].toFloat() }
        return normalizeRow(row)
    }

    private fun readUInt32LittleEndian(bytes: ByteArray): Long =
        (bytes[0].toLong() and 0xFF) or
            ((bytes[1].toLong() and 0xFF) shl 8) or
            ((bytes[2].toLong() and 0xFF) shl 16) or
            ((bytes[3].toLong() and 0xFF) shl 24)

    override suspend fun embedDocuments(texts: List<String>): List<FloatArray> =
        texts.map { embedText(it) }

    override suspend fun embedQuery(text: String): FloatArray = embedText(text)
}
