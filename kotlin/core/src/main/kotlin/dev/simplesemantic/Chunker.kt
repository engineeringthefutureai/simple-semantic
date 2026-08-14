package dev.simplesemantic

/**
 * A slice of a source text, with the index it came from.
 */
public data class Chunk(val index: Int, val text: String)

/**
 * Fixed window over code points, with a fixed overlap.
 *
 * The chunker's id is part of the content hash (SPEC.md §4.1), so changing the
 * strategy correctly invalidates every stored vector.
 *
 * Boundaries are counted in **Unicode code points**, not UTF-16 code units.
 * Counting `String.length` would make the JVM disagree with Python on any text
 * containing an emoji or a rarer CJK character — one code point, two UTF-16
 * units — and would sometimes split a surrogate pair, producing a chunk that is
 * not valid text.
 *
 * Deliberately not sentence- or token-aware. A smarter chunker is a real
 * improvement to retrieval quality and a real source of cross-language
 * divergence; this project's claim is about the index, so the chunker stays
 * dumb and identical.
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
