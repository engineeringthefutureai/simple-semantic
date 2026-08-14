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
            val raw = CanonicalJson.parseObject(Files.readString(path, Charsets.UTF_8))

            for (key in listOf("embedder_id", "dimension", "documents", "queries")) {
                if (!raw.containsKey(key)) {
                    throw SimpleSemanticException("$path: fixture is missing '$key'")
                }
            }

            val dimension = (raw["dimension"] as? Long)?.toInt()
                ?: throw SimpleSemanticException("$path: 'dimension' is not an integer")
            val embedderId = raw["embedder_id"] as? String
                ?: throw SimpleSemanticException("$path: 'embedder_id' is not a string")
            if (!embedderId.endsWith("@$dimension")) {
                // Repeated at load time: a fixture can be hand-edited after generation.
                throw SimpleSemanticException(
                    "$path: embedder_id '$embedderId' disagrees with dimension $dimension",
                )
            }

            fun load(section: String): Map<String, FloatArray> {
                val entries = raw[section] as? List<*>
                    ?: throw SimpleSemanticException("$path: '$section' is not an array")
                val out = HashMap<String, FloatArray>(entries.size)
                for (entry in entries) {
                    @Suppress("UNCHECKED_CAST")
                    val record = entry as Map<String, Any?>
                    val values = record["vector"] as? List<*>
                        ?: throw SimpleSemanticException("$path: $section entry has no vector")
                    if (values.size != dimension) {
                        throw SimpleSemanticException(
                            "$path: $section entry ${record["id"] ?: record["key"]} has " +
                                "${values.size} dimensions, expected $dimension",
                        )
                    }
                    out[record["key"] as String] =
                        FloatArray(values.size) { i -> (values[i] as Number).toFloat() }
                }
                return out
            }

            return ReplayEmbedder(
                id = embedderId,
                dimension = dimension,
                documents = load("documents"),
                queries = load("queries"),
                producesNormalized = raw["normalized"] as? Boolean ?: false,
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
