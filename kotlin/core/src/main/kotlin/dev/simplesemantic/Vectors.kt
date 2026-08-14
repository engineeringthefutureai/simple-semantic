package dev.simplesemantic

import java.lang.foreign.Arena
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.nio.file.Path
import java.nio.file.StandardOpenOption

/**
 * `vectors.f32` — the dense matrix. SPEC.md §3.
 *
 * Mapped with the Foreign Function & Memory API rather than `ByteBuffer`.
 * `ByteBuffer` is capped at `Integer.MAX_VALUE` bytes, which at 3072 dimensions
 * is about 175,000 rows — a ceiling a real knowledge base reaches, and one that
 * forces a redesign rather than a patch when it does. `MemorySegment` maps past
 * it.
 */
public class VectorStore private constructor(
    private val arena: Arena,
    private val segment: MemorySegment,
    public val rowCount: Int,
    public val dimension: Int,
) : AutoCloseable {

    public companion object {
        /**
         * Little-endian float32, explicitly.
         *
         * `ValueLayout.JAVA_FLOAT` uses the platform byte order, which is
         * correct on x86 and silently wrong elsewhere — and JVM `ByteBuffer`
         * defaults to big-endian outright. The format is little-endian, so the
         * order is named here rather than inherited. `_UNALIGNED` because a
         * mapped file offers no alignment guarantee the JVM will accept.
         */
        public val F32: ValueLayout.OfFloat =
            ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN)

        /** Little-endian uint64, for `offsets.bin`. SPEC.md §5. */
        public val U64: ValueLayout.OfLong =
            ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN)

        public fun open(path: Path, rowCount: Int, dimension: Int): VectorStore {
            val expected = rowCount.toLong() * dimension * 4
            val actual = java.nio.file.Files.size(path)
            if (actual != expected) {
                throw CorruptIndexException(
                    "$path is $actual bytes but the manifest declares row_count $rowCount " +
                        "x dimension $dimension x 4 = $expected. The index is truncated or " +
                        "was copied while being written.",
                )
            }
            val arena = Arena.ofShared()
            if (rowCount == 0) {
                return VectorStore(arena, MemorySegment.NULL, 0, dimension)
            }
            FileChannel.open(path, StandardOpenOption.READ).use { channel ->
                val segment = channel.map(FileChannel.MapMode.READ_ONLY, 0, expected, arena)
                return VectorStore(arena, segment, rowCount, dimension)
            }
        }
    }

    /** Copy one row out as a `FloatArray`. */
    public fun row(index: Int): FloatArray {
        require(index in 0 until rowCount) { "row $index out of range for $rowCount rows" }
        val out = FloatArray(dimension)
        var offset = index.toLong() * dimension * 4
        for (i in 0 until dimension) {
            out[i] = segment.get(F32, offset)
            offset += 4
        }
        return out
    }

    /**
     * Cosine similarity of row [index] with an already-normalized [query].
     *
     * A plain sequential loop, accumulating in `Double`. C2 auto-vectorizes
     * this acceptably; the incubating Vector API would buy a constant factor at
     * the cost of forcing `--add-modules jdk.incubator.vector` on every
     * consumer of the library, which is not a trade worth making without a JMH
     * number saying otherwise.
     *
     * Accumulation is in `Double` because a `Float` accumulator loses enough
     * precision at 3072 dimensions to reorder near-ties (SPEC.md §8).
     */
    public fun dot(index: Int, query: FloatArray): Double {
        require(query.size == dimension) {
            "query has ${query.size} dimensions, index has $dimension"
        }
        var sum = 0.0
        var offset = index.toLong() * dimension * 4
        for (i in 0 until dimension) {
            sum += segment.get(F32, offset).toDouble() * query[i].toDouble()
            offset += 4
        }
        return sum
    }

    override fun close() {
        arena.close()
    }
}

/**
 * L2 normalization, exactly as SPEC.md §3.1 specifies.
 *
 * The arithmetic is pinned rather than left to the language because the two
 * implementations must produce byte-identical files: sequential accumulation in
 * `Double`, ascending index, correctly-rounded `sqrt`, then a single narrowing
 * to `Float`. A pairwise or blocked sum — which any vectorized reduction is
 * free to use — differs in the last ulp of the `Double`, which is enough to
 * change the final `Float` rounding.
 */
public fun normalizeRow(vector: FloatArray): FloatArray {
    var sumOfSquares = 0.0
    for (i in vector.indices) {
        val value = vector[i].toDouble()
        sumOfSquares += value * value
    }
    val norm = Math.sqrt(sumOfSquares)
    val out = FloatArray(vector.size)
    if (norm == 0.0) {
        // A zero row has no direction, and 0/0 would put a NaN in the matrix
        // that silently poisons every later dot product. e_0 is arbitrary but
        // defined: a valid unit vector that simply ranks poorly.
        if (out.isNotEmpty()) out[0] = 1.0f
        return out
    }
    for (i in vector.indices) {
        out[i] = (vector[i].toDouble() / norm).toFloat()
    }
    return out
}

/** Serialize a row to little-endian float32 bytes. SPEC.md §3. */
public fun FloatArray.toLittleEndianBytes(): ByteArray {
    val bytes = ByteArray(size * 4)
    for (i in indices) {
        val bits = java.lang.Float.floatToRawIntBits(this[i])
        val base = i * 4
        bytes[base] = (bits and 0xFF).toByte()
        bytes[base + 1] = ((bits ushr 8) and 0xFF).toByte()
        bytes[base + 2] = ((bits ushr 16) and 0xFF).toByte()
        bytes[base + 3] = ((bits ushr 24) and 0xFF).toByte()
    }
    return bytes
}
