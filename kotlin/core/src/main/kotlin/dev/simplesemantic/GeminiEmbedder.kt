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
 * Built on the JDK's `java.net.http.HttpClient` rather than a client library.
 *
 * Two model quirks that are silent when you get them wrong:
 * `gemini-embedding-001` pre-normalizes only its default 3072-dimension output,
 * and `gemini-embedding-002` aggregates a multi-input request into one
 * embedding unless each input is wrapped individually. This class normalizes
 * unconditionally and always wraps individually, which is correct for both.
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

    /** 001 only pre-normalizes at 3072; the index normalizes either way. */
    override val producesNormalized: Boolean = dimension == PRE_NORMALIZED_DIMENSION

    private val client: HttpClient = HttpClient.newBuilder().connectTimeout(timeout).build()

    override suspend fun embedDocuments(texts: List<String>): List<FloatArray> {
        if (texts.isEmpty()) return emptyList()
        return request(texts, taskType = "RETRIEVAL_DOCUMENT").map { normalizeRow(it) }
    }

    // Separate from embedDocuments: the task type genuinely differs.
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
                // Wrapped individually — see the note above.
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
        val parsed = try {
            WireJson.decodeFromString(EmbedResponseWire.serializer(), body)
        } catch (exc: kotlinx.serialization.SerializationException) {
            throw SimpleSemanticException("$model returned unreadable JSON: ${exc.message}")
        }
        val embeddings = parsed.embeddings
        if (embeddings == null || embeddings.size != texts.size) {
            throw SimpleSemanticException(
                "$model returned ${embeddings?.size ?: "null"} embeddings for ${texts.size} " +
                    "inputs. If this says 1, the API aggregated the batch into a single vector.",
            )
        }
        return embeddings.map { embedding ->
            if (embedding.values.size != dimension) {
                throw SimpleSemanticException(
                    "$model returned ${embedding.values.size} dimensions, expected $dimension",
                )
            }
            embedding.values.toFloatArray()
        }
    }

    /**
     * Retry on 429 and 5xx with exponential backoff. Retry belongs to the
     * embedder; the index has no idea what a rate limit is.
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
