package dev.simplesemantic

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive

/**
 * The shapes this project reads, declared once.
 *
 * Encoding stays hand-written — SPEC.md §7 pins a canonical form no library
 * guarantees by default — but decoding has no such constraint, so the wire
 * shapes are data classes rather than casts out of a `Map<String, Any?>`.
 */
internal val WireJson: Json = Json { ignoreUnknownKeys = true }

/**
 * Just enough of the manifest to check the version.
 *
 * SPEC.md §2.2 makes `format_version` check 1, before anything structural: a
 * later version is free to change the field set, and decoding the whole
 * manifest first would report that as corruption rather than as a version the
 * reader is too old to understand.
 */
@Serializable
internal data class ManifestVersionWire(@SerialName("format_version") val formatVersion: Int)

/** `manifest.json`. SPEC.md §2. */
@Serializable
internal data class ManifestWire(
    @SerialName("format_version") val formatVersion: Int,
    @SerialName("embedder_id") val embedderId: String,
    val dimension: Int,
    val normalized: Boolean = true,
    @SerialName("chunker_id") val chunkerId: String,
    @SerialName("hash_algorithm") val hashAlgorithm: String = "sha256",
    @SerialName("row_count") val rowCount: Int,
    @SerialName("live_count") val liveCount: Int,
    @SerialName("created_at") val createdAt: String,
    @SerialName("updated_at") val updatedAt: String,
)

/** One line of `docs.jsonl`. SPEC.md §4. */
@Serializable
internal data class DocumentWire(
    val id: String,
    val text: String,
    val meta: JsonObject = JsonObject(emptyMap()),
    val hash: String = "",
)

/** The replay fixture. SPEC.md appendix B. */
@Serializable
internal data class FixtureWire(
    @SerialName("embedder_id") val embedderId: String,
    val dimension: Int,
    val normalized: Boolean = false,
    val documents: List<FixtureRecordWire>,
    val queries: List<FixtureRecordWire>,
)

@Serializable
internal data class FixtureRecordWire(
    val key: String,
    // Double, not Float: Python reads these decimals into float64 and narrows,
    // so Kotlin must round twice as well. Parsing straight to Float rounds once,
    // and single rounding is not always equal to double rounding.
    val vector: List<Double>,
    val id: String = "",
    val file: String = "",
    val prompt: String = "",
    val target: String = "",
)

/** A `batchEmbedContents` response. */
@Serializable
internal data class EmbedResponseWire(val embeddings: List<EmbeddingWire>? = null)

@Serializable
internal data class EmbeddingWire(val values: List<Float>)

/**
 * Parse one JSONL input line: `{"id": ..., "text": ..., "meta": {...}}`.
 *
 * The public entry point for callers holding JSONL — the CLI, mainly — so that
 * no consumer has to depend on the JSON library or reproduce the `meta`
 * conversion.
 */
public object DocumentJson {
    public fun parseLine(line: String): Document {
        val wire = try {
            WireJson.decodeFromString(DocumentWire.serializer(), line)
        } catch (exc: kotlinx.serialization.SerializationException) {
            throw JsonException(exc.message ?: "invalid document JSON")
        }
        return Document(wire.id, wire.text, wire.meta.toMetaMap())
    }
}

/**
 * `meta` as the public API sees it.
 *
 * The format restricts values to strings, booleans, null, 64-bit integers and
 * nestings of those (SPEC.md §7.2), so the conversion is total.
 */
internal fun JsonObject.toMetaMap(): Map<String, Any?> =
    mapValues { (_, value) -> value.toKotlinValue() }

private fun JsonElement.toKotlinValue(): Any? = when (this) {
    // JsonNull is itself a JsonPrimitive, so it has to be matched first.
    is JsonNull -> null
    is JsonPrimitive -> when {
        isString -> content
        content == "true" -> true
        content == "false" -> false
        else -> content.toLongOrNull() ?: content.toDouble()
    }
    is JsonArray -> map { it.toKotlinValue() }
    is JsonObject -> toMetaMap()
}
