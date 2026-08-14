package dev.simplesemantic

/**
 * A slice of a source text, with the index it came from.
 */
public data class Chunk(val index: Int, val text: String)

/**
 * Fixed window over code points, with a fixed overlap.
 *
 * The chunker's id is part of the content hash (SPEC.md §4.1), so changing the
 * strategy invalidates every stored vector.
 *
 * Boundaries are counted in Unicode code points, not UTF-16 code units:
 * `String.length` would disagree with Python above the BMP and would sometimes
 * split a surrogate pair.
 *
 * Deliberately not sentence- or token-aware: a smarter chunker is a source of
 * cross-language divergence.
 */
public class FixedChunker(
    public val size: Int = 512,
    public val overlap: Int = 64,
) {
    init {
        require(size > 0) { "chunk size must be positive, got $size" }
        require(overlap in 0 until size) {
            "overlap must be in [0, size), got $overlap with size $size"
        }
    }

    public val id: String = "fixed-$size-overlap-$overlap"

    public fun chunk(text: String): List<Chunk> {
        if (text.isBlank()) return emptyList()

        val points = text.codePoints().toArray()
        val step = size - overlap
        val chunks = ArrayList<Chunk>()
        var start = 0
        var index = 0
        while (start < points.size) {
            val end = minOf(start + size, points.size)
            chunks.add(Chunk(index, String(points, start, end - start)))
            index++
            if (start + size >= points.size) break
            start += step
        }
        return chunks
    }
}
