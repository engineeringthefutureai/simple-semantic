package dev.simplesemantic

import java.security.MessageDigest
import java.time.Instant
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter

public const val FORMAT_VERSION: Int = 1

public object FileNames {
    public const val MANIFEST: String = "manifest.json"
    public const val VECTORS: String = "vectors.f32"
    public const val DOCS: String = "docs.jsonl"
    public const val OFFSETS: String = "offsets.bin"
    public const val TOMBSTONES: String = "tombstones.bits"
}

private val RFC3339: DateTimeFormatter =
    DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss'Z'").withZone(ZoneOffset.UTC)

/** RFC 3339, UTC, second precision, Z suffix. SPEC.md §2. */
public fun utcNow(): String = RFC3339.format(Instant.now())

/**
 * SPEC.md §4.1: `sha256(text || 0x00 || embedder_id || 0x00 || chunker_id)`.
 *
 * The NUL separators are unambiguous because none of the three inputs may
 * contain a NUL byte.
 */
public fun contentHash(text: String, embedderId: String, chunkerId: String): String {
    val digest = MessageDigest.getInstance("SHA-256")
    digest.update(text.toByteArray(Charsets.UTF_8))
    digest.update(0)
    digest.update(embedderId.toByteArray(Charsets.UTF_8))
    digest.update(0)
    digest.update(chunkerId.toByteArray(Charsets.UTF_8))
    return digest.digest().toHex()
}

internal fun ByteArray.toHex(): String {
    val out = StringBuilder(size * 2)
    for (byte in this) {
        val value = byte.toInt() and 0xFF
        out.append(HEX_DIGITS[value ushr 4]).append(HEX_DIGITS[value and 0x0F])
    }
    return out.toString()
}

private const val HEX_DIGITS = "0123456789abcdef"

/** The header for the other three files. SPEC.md §2. */
public data class Manifest(
    val embedderId: String,
    val dimension: Int,
    val chunkerId: String,
    val rowCount: Int,
    val liveCount: Int,
    val createdAt: String,
    val updatedAt: String,
    val formatVersion: Int = FORMAT_VERSION,
    val normalized: Boolean = true,
    val hashAlgorithm: String = "sha256",
) {
    /** Key order is fixed by SPEC.md §2, and fixed here. */
    public fun encode(): String = CanonicalJson.encodeObject(
        linkedMapOf(
            "format_version" to formatVersion,
            "embedder_id" to embedderId,
            "dimension" to dimension,
            "normalized" to normalized,
            "chunker_id" to chunkerId,
            "hash_algorithm" to hashAlgorithm,
            "row_count" to rowCount,
            "live_count" to liveCount,
            "created_at" to createdAt,
            "updated_at" to updatedAt,
        ),
    )

    public companion object {
        public fun decode(raw: String, path: String): Manifest {
            // Version first: a v2 manifest may not carry v1's fields at all, and
            // "too old to read this" is the useful error, not "corrupt".
            val version = try {
                WireJson.decodeFromString(ManifestVersionWire.serializer(), raw).formatVersion
            } catch (exc: kotlinx.serialization.SerializationException) {
                throw CorruptIndexException("$path: manifest is not valid JSON (${exc.message})")
            }
            if (version != FORMAT_VERSION) {
                throw FormatVersionException(path, version, FORMAT_VERSION)
            }

            val wire = try {
                WireJson.decodeFromString(ManifestWire.serializer(), raw)
            } catch (exc: kotlinx.serialization.SerializationException) {
                throw CorruptIndexException("$path: manifest is not valid JSON (${exc.message})")
            }
            return Manifest(
                embedderId = wire.embedderId,
                dimension = wire.dimension,
                chunkerId = wire.chunkerId,
                rowCount = wire.rowCount,
                liveCount = wire.liveCount,
                createdAt = wire.createdAt,
                updatedAt = wire.updatedAt,
                formatVersion = wire.formatVersion,
                normalized = wire.normalized,
                hashAlgorithm = wire.hashAlgorithm,
            )
        }
    }
}

/**
 * A bit per row, LSB-first within each byte. SPEC.md §6.
 *
 * MSB-first produces a file of the same size that parses without error and
 * disagrees about which rows are deleted.
 */
public class Tombstones private constructor(
    private var bits: ByteArray,
    private var rows: Int,
) {
    public companion object {
        public fun empty(rowCount: Int): Tombstones =
            Tombstones(ByteArray(bytesFor(rowCount)), rowCount)

        public fun of(raw: ByteArray, rowCount: Int): Tombstones {
            val needed = bytesFor(rowCount)
            if (raw.size != needed) {
                throw CorruptIndexException(
                    "${FileNames.TOMBSTONES} is ${raw.size} bytes but row_count $rowCount " +
                        "requires exactly $needed",
                )
            }
            return Tombstones(raw.copyOf(), rowCount)
        }

        private fun bytesFor(rowCount: Int): Int = (rowCount + 7) / 8
    }

    public val rowCount: Int get() = rows

    public var deletedCount: Int = bits.sumOf { Integer.bitCount(it.toInt() and 0xFF) }
        private set

    public fun isDeleted(row: Int): Boolean {
        require(row in 0 until rows) { "row $row out of range for $rows rows" }
        return (bits[row ushr 3].toInt() and (1 shl (row and 7))) != 0
    }

    /** Set the bit. Returns true if this call changed it. */
    public fun markDeleted(row: Int): Boolean {
        if (isDeleted(row)) return false
        bits[row ushr 3] = (bits[row ushr 3].toInt() or (1 shl (row and 7))).toByte()
        deletedCount++
        return true
    }

    public fun growTo(rowCount: Int) {
        require(rowCount >= rows) { "cannot shrink tombstones from $rows to $rowCount" }
        rows = rowCount
        val needed = bytesFor(rowCount)
        if (bits.size < needed) bits = bits.copyOf(needed)
    }

    public fun anyDeleted(): Boolean = deletedCount > 0

    /** Boolean array, true where the row is live. */
    public fun liveMask(): BooleanArray = BooleanArray(rows) { !isDeleted(it) }

    // Padding bits stay zero: markDeleted range-checks, growTo appends zeros.
    public fun toByteArray(): ByteArray = bits.copyOf()
}
