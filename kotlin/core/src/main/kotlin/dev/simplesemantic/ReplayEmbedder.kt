package dev.simplesemantic

import java.nio.file.Files
import java.nio.file.Path
import java.security.MessageDigest

/**
 * The text asked for is not in the recording.
 *
 * Names the mode as well as the text, because the commonest cause is asking for
 * a document embedding of something recorded only as a query. Those are
 * genuinely different vectors — different task types — and serving one for the
 * other is a silent quality loss.
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
 * Serve recorded vectors instead of calling a model.
 *
 * The third embedder, and the one that closes a real gap. [HashingEmbedder] is
 * deterministic but semantically meaningless: it can prove the *format* is
 * correct and nothing about whether search retrieves. [GeminiEmbedder]
 * retrieves properly but needs a credential, a network and money, so no test
 * suite can depend on it.
 *
 * This one reads vectors a real model produced once, recorded to a JSON fixture
 * that both implementations parse. That makes genuine retrieval-quality
 * assertions runnable in CI, offline, with identical numbers every run — and it
 * lets conformance prove byte-identity from *real* embeddings rather than only
 * from the hashing embedder.
 *
 * A miss is a loud error, never a zero vector or a fallback. A mock that
 * silently invents a plausible answer is the failure mode this whole project is
 * built to refuse.
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
                // The same guard build_fixture.py applies, repeated at load time
                // because a fixture can be hand-edited after it is generated.
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
        // Copy: a caller mutating the result must not corrupt the recording.
        return vector.copyOf()
    }

    override suspend fun embedDocuments(texts: List<String>): List<FloatArray> =
        texts.map { lookup("document", documents, it) }

    // Deliberately a different table from embedDocuments. The recording was made
    // with RETRIEVAL_QUERY here and RETRIEVAL_DOCUMENT there, so the same string
    // has two different correct answers.
    override suspend fun embedQuery(text: String): FloatArray = lookup("query", queries, text)
}
