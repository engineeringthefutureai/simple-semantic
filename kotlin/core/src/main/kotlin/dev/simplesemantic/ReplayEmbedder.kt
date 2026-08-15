package dev.simplesemantic

import java.nio.file.Files
import java.nio.file.Path
import java.security.MessageDigest

/**
 * The text asked for is not in the recording. Names the mode too: the commonest
 * cause is asking for a document embedding of something recorded as a query.
 */
public class ReplayMissException(
    mode: String,
    text: String,
    key: String,
    source: String,
) : SimpleSemanticException(
    "no recorded $mode embedding for '${text.take(57).let { if (text.length > 60) "$it..." else it }}' " +
        "(sha256 ${key.take(16)}...) in $source. Either the text changed since the " +
        "fixture was generated, or it was recorded as a " +
        "${if (mode == "document") "query" else "document"} instead.",
)

/**
 * Serve recorded vectors instead of calling a model. SPEC.md appendix B.
 *
 * Reads vectors a real model produced once, recorded to a JSON fixture both
 * implementations parse. That makes retrieval-quality assertions runnable
 * offline with identical numbers every run, and lets conformance prove
 * byte-identity from real embeddings.
 *
 * A miss is a loud error, never a zero vector: inventing a plausible answer
 * would turn "not in the recording" into "retrieval quietly got worse".
 */
public class ReplayEmbedder(
    override val id: String,
    override val dimension: Int,
    private val documents: Map<String, FloatArray>,
    private val queries: Map<String, FloatArray>,
    override val producesNormalized: Boolean = false,
    private val source: String = "<memory>",
) : Embedder {

    /** No network, so batching exists only to satisfy the interface. */
    override val maxBatchSize: Int = 1024

    public val size: Int get() = documents.size + queries.size

    public companion object {
        /** Load a fixture written by `conformance/stories/build_fixture.py`. */
        public fun fromFile(path: Path): ReplayEmbedder {
            val wire = try {
                WireJson.decodeFromString(
                    FixtureWire.serializer(),
                    Files.readString(path, Charsets.UTF_8),
                )
            } catch (exc: kotlinx.serialization.SerializationException) {
                throw SimpleSemanticException("$path: not a valid fixture (${exc.message})")
            }

            if (!wire.embedderId.endsWith("@${wire.dimension}")) {
                // Repeated at load time: a fixture can be hand-edited after generation.
                throw SimpleSemanticException(
                    "$path: embedder_id '${wire.embedderId}' disagrees with " +
                        "dimension ${wire.dimension}",
                )
            }

            fun vectors(section: String, records: List<FixtureRecordWire>) =
                records.associate { record ->
                    if (record.vector.size != wire.dimension) {
                        throw SimpleSemanticException(
                            "$path: $section entry ${record.id.ifEmpty { record.key }} has " +
                                "${record.vector.size} dimensions, expected ${wire.dimension}",
                        )
                    }
                    record.key to FloatArray(record.vector.size) {
                        record.vector[it].toFloat()
                    }
                }

            return ReplayEmbedder(
                id = wire.embedderId,
                dimension = wire.dimension,
                documents = vectors("documents", wire.documents),
                queries = vectors("queries", wire.queries),
                producesNormalized = wire.normalized,
                source = path.toString(),
            )
        }

        /** sha256 of the exact text that was embedded. Must match build_fixture.py. */
        public fun contentKey(text: String): String =
            MessageDigest.getInstance("SHA-256").digest(text.toByteArray(Charsets.UTF_8)).toHex()
    }

    private fun lookup(mode: String, table: Map<String, FloatArray>, text: String): FloatArray {
        val key = contentKey(text)
        val vector = table[key] ?: throw ReplayMissException(mode, text, key, source)
        // Copy: a caller must not be able to mutate the recording.
        return vector.copyOf()
    }

    override suspend fun embedDocuments(texts: List<String>): List<FloatArray> =
        texts.map { lookup("document", documents, it) }

    // A different table from embedDocuments: RETRIEVAL_QUERY here,
    // RETRIEVAL_DOCUMENT there, so the same string has two right answers.
    override suspend fun embedQuery(text: String): FloatArray = lookup("query", queries, text)
}
