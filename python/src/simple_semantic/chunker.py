"""Fixed-size chunking with overlap.

The chunker's id is part of the content hash (SPEC.md §4.1), so changing the
strategy invalidates every stored vector.

Boundaries are counted in Unicode code points: UTF-16 code units would make the
JVM and Python disagree on anything above the BMP.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """A slice of a source text, with the index it came from."""

    index: int
    text: str


class FixedChunker:
    """Fixed window over code points, with a fixed overlap.

    Deliberately not sentence- or token-aware: a smarter chunker is a source of
    cross-language divergence.
    """

    def __init__(self, size: int = 512, overlap: int = 64) -> None:
        if size <= 0:
            raise ValueError(f"chunk size must be positive, got {size}")
        if not 0 <= overlap < size:
            raise ValueError(f"overlap must be in [0, size), got {overlap} with size {size}")
        self.size = size
        self.overlap = overlap
        self.id = f"fixed-{size}-overlap-{overlap}"

    def chunk(self, text: str) -> list[Chunk]:
        """Split ``text``. Empty and whitespace-only input yields no chunks.

        Keeps zero vectors out of the index; the e_0 fallback in SPEC.md §3.1 is
        the safety net, not the plan.
        """
        if not text.strip():
            return []
        # Explicit code points, matching how the Kotlin side must walk the string.
        points = list(text)
        step = self.size - self.overlap
        chunks: list[Chunk] = []
        start = 0
        index = 0
        while start < len(points):
            piece = "".join(points[start : start + self.size])
            chunks.append(Chunk(index=index, text=piece))
            index += 1
            if start + self.size >= len(points):
                break
            start += step
        return chunks
