package dev.simplesemantic

/** Errors. Each carries the specific ids, paths or sizes involved. */
public open class SimpleSemanticException(message: String, cause: Throwable? = null) :
    RuntimeException(message, cause)

/**
 * The index was written by a different embedder than the one configured.
 * SPEC.md §2.1.
 */
public class EmbedderMismatchException(
    public val path: String,
    public val indexEmbedderId: String,
    public val configuredEmbedderId: String,
) : SimpleSemanticException(
    "index at $path was written with embedder_id '$indexEmbedderId' but the " +
        "configured embedder is '$configuredEmbedderId'. Scores across two " +
        "embedding spaces are meaningless. Re-index with the new embedder, or " +
        "open with the original one.",
)

public class FormatVersionException(path: String, found: Any?, supported: Int) :
    SimpleSemanticException(
        "index at $path declares format_version $found; this build supports version $supported",
    )

/** A structural check from SPEC.md §2.2 failed. */
public class CorruptIndexException(message: String) : SimpleSemanticException(message)

/** A meta value is outside the types the format can round-trip. SPEC.md §7.2. */
public class MetaValueException(keyPath: String, value: Any?) : SimpleSemanticException(
    "meta value at '$keyPath' has unsupported type ${value?.let { it::class.simpleName }} " +
        "($value). Permitted: String, Boolean, null, 64-bit integer, List, Map with " +
        "String keys. Floating-point values are not permitted in format v1 — store " +
        "the value as a String.",
)

/** A class could not be turned into indexable documents. */
public class ExtractionException(message: String) : SimpleSemanticException(message)

/** Input was not valid JSON. */
public class JsonException(message: String) : SimpleSemanticException(message)
