package dev.simplesemantic

import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.time.Duration
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext

/**
 * The one component that makes a network call.
 *
 * Built on `java.net.http.HttpClient` from the JDK rather than a client
 * library, so that `core` keeps zero runtime dependencies beyond the Kotlin
 * stdlib and coroutines.
 *
 * Model quirks worth stating in code rather than in a wiki nobody reads:
 *
 * - `gemini-embedding-001` pre-normalizes **only** its default 3072-dimension
 *   output. Any smaller `outputDimensionality` comes back unnormalized and must
 *   be normalized by hand — which this class does unconditionally, and which
 *   the index then does again at write time.
 * - `gemini-embedding-002` does normalize truncated output, but aggregates
 *   multiple inputs in a single request into one embedding unless each input is
 *   wrapped individually. A batch loop written against `001` therefore gets one
 *   vector where it expected N, silently, against `002`. This class always
 *   wraps inputs individually, which is correct for both.
 */
public class GeminiEmbedder(
    private val model: String = "gemini-embedding-001",
    override val dimension: Int = 768,
    apiKey: String? = null,
    override val maxBatchSize: Int = 100,
    private val timeout: Duration = Duration.ofSeconds(60),
    private val maxRetries: Int = 5,
) : Embedder {

    private val apiKey: String = apiKey
        ?: System.getenv("GEMINI_API_KEY")
        ?: throw IllegalArgumentException(
            "no API key: pass apiKey or set GEMINI_API_KEY. Use HashingEmbedder for " +
                "tests — it needs neither.",
        )

    override val id: String = "$model@$dimension"

    /** Reported honestly: 001 only pre-normalizes at 3072. */
    override val producesNormalized: Boolean = dimension == PRE_NORMALIZED_DIMENSION

    private val client: HttpClient = HttpClient.newBuilder().connectTimeout(timeout).build()

    override suspend fun embedDocuments(texts: List<String>): List<FloatArray> {
        if (texts.isEmpty()) return emptyList()
        return request(texts, taskType = "RETRIEVAL_DOCUMENT").map { normalizeRow(it) }
    }

    // Separate from embedDocuments because the task type genuinely differs.
    // Collapsing these into one embed() is a silent quality loss that no test
    // catches unless you already know to look for it.
    override suspend fun embedQuery(text: String): FloatArray =
        normalizeRow(request(listOf(text), taskType = "RETRIEVAL_QUERY").single())

    private suspend fun request(texts: List<String>, taskType: String): List<FloatArray> {
        require(texts.size <= maxBatchSize) {
            "batch of ${texts.size} exceeds maxBatchSize $maxBatchSize; the index batches " +
                "for you, so this is a caller bug"
        }

        val payload = buildString {
            append("{\"requests\":[")
            texts.forEachIndexed { i, text ->
                if (i > 0) append(',')
                append("{\"model\":\"models/").append(model).append("\",")
                // Each input wrapped individually — see the note about
                // gemini-embedding-002 aggregating multiple parts.
                append("\"content\":{\"parts\":[{\"text\":")
                append(CanonicalJson.encodeString(text))
                append("}]},")
                append("\"taskType\":\"").append(taskType).append("\",")
                append("\"outputDimensionality\":").append(dimension)
                append('}')
            }
            append("]}")
        }

        val body = sendWithRetry(payload)
        val parsed = CanonicalJson.parseObject(body)
        val embeddings = parsed["embeddings"] as? List<*>
        if (embeddings == null || embeddings.size != texts.size) {
            val got = embeddings?.size?.toString() ?: "null"
            throw SimpleSemanticException(
                "$model returned $got embeddings for ${texts.size} inputs. If this says 1, " +
                    "the API aggregated the batch into a single vector.",
            )
        }
        return embeddings.map { entry ->
            @Suppress("UNCHECKED_CAST")
            val values = (entry as Map<String, Any?>)["values"] as List<*>
            if (values.size != dimension) {
                throw SimpleSemanticException(
                    "$model returned ${values.size} dimensions, expected $dimension",
                )
            }
            FloatArray(values.size) { i -> (values[i] as Number).toFloat() }
        }
    }

    /**
     * Retry on 429 and 5xx with exponential backoff.
     *
     * Retry belongs to the embedder, not to the index: the index has no idea
     * what a rate limit is and should not grow one.
     */
    private suspend fun sendWithRetry(payload: String): String {
        val request = HttpRequest.newBuilder(URI.create(ENDPOINT.format(model)))
            .header("content-type", "application/json")
            .header("x-goog-api-key", apiKey)
            .timeout(timeout)
            .POST(HttpRequest.BodyPublishers.ofString(payload))
            .build()

        var delayMillis = 1000L
        var last: String = "no attempt made"
        repeat(maxRetries) { attempt ->
            val response = try {
                withContext(Dispatchers.IO) {
                    client.send(request, HttpResponse.BodyHandlers.ofString())
                }
            } catch (exc: java.io.IOException) {
                last = "transport failure: ${exc.message}"
                null
            }
            if (response != null) {
                if (response.statusCode() < 400) return response.body()
                val detail = "HTTP ${response.statusCode()}: ${response.body().take(400)}"
                if (response.statusCode() != 429 && response.statusCode() < 500) {
                    throw SimpleSemanticException(detail)
                }
                last = detail
            }
            if (attempt < maxRetries - 1) {
                delay(delayMillis)
                delayMillis *= 2
            }
        }
        throw SimpleSemanticException(
            "embedding request failed after $maxRetries attempts: $last",
        )
    }

    private companion object {
        const val ENDPOINT =
            "https://generativelanguage.googleapis.com/v1beta/models/%s:batchEmbedContents"
        const val PRE_NORMALIZED_DIMENSION = 3072
    }
}
