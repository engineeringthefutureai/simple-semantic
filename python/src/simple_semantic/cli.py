"""``simple-semantic`` command line: index, search, stats, compact."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .chunker import FixedChunker
from .embedder import Embedder, HashingEmbedder
from .errors import SimpleSemanticError
from .index import Document, SemanticIndex
from .replay import ReplayEmbedder
from .wire import DocumentWire, decode


def _build_embedder(args: argparse.Namespace) -> Embedder:
    if args.embedder == "hashing":
        return HashingEmbedder(dimension=args.dimension, seed=args.seed)
    if args.embedder == "replay":
        if not args.fixture:
            raise SimpleSemanticError("--embedder replay needs --fixture PATH")
        # Dimension and id come from the fixture, not the flags.
        return ReplayEmbedder.from_file(args.fixture)
    if args.embedder == "gemini":
        from .gemini import GeminiEmbedder

        return GeminiEmbedder(model=args.model, dimension=args.dimension)
    raise SimpleSemanticError(f"unknown embedder {args.embedder!r}")


def _read_documents(source: Path | None) -> list[Document]:
    """Read JSONL documents from a file or stdin.

    Each line is ``{"id": ..., "text": ..., "meta": {...}}``; ``meta`` optional.
    """
    # Not a context manager: the handle is either stdin, which must not be
    # closed, or a file this function owns and closes in the finally block.
    handle = sys.stdin if source is None else open(source, encoding="utf-8")  # noqa: SIM115
    try:
        documents = []
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                wire = decode(DocumentWire, json.loads(line), f"line {number}")
            except ValueError as exc:
                raise SimpleSemanticError(f"line {number}: not valid JSON ({exc})") from exc
            documents.append(Document(id=wire.id, text=wire.text, meta=dict(wire.meta)))
        return documents
    finally:
        if source is not None:
            handle.close()


async def _cmd_index(args: argparse.Namespace) -> int:
    embedder = _build_embedder(args)
    documents = _read_documents(args.input)

    if args.chunk:
        chunker = FixedChunker(size=args.chunk_size, overlap=args.chunk_overlap)
        chunker_id = chunker.id
        chunked = [
            Document(id=f"{doc.id}#{piece.index}", text=piece.text, meta=doc.meta)
            for doc in documents
            for piece in chunker.chunk(doc.text)
        ]
        documents = chunked
    else:
        chunker_id = "none"

    path = Path(args.path)
    if (path / "manifest.json").exists():
        index = SemanticIndex.open(path, embedder)
    else:
        index = SemanticIndex.create(path, embedder, chunker_id=chunker_id)

    with index:
        result = await index.add_all(documents)
        print(
            f"added {result.added}, replaced {result.replaced}, "
            f"skipped {result.skipped} (unchanged) -> {index.live_count()} live rows"
        )
    return 0


async def _cmd_search(args: argparse.Namespace) -> int:
    embedder = _build_embedder(args)
    with SemanticIndex.open(Path(args.path), embedder) as index:
        results = await index.search(args.query, k=args.k)

    if args.json:
        for rank, result in enumerate(results, start=1):
            print(
                json.dumps(
                    {
                        "rank": rank,
                        "id": result.id,
                        # Fixed precision as a string; see SPEC.md §7.2.
                        "score": f"{result.score:.12f}",
                        "row": result.row,
                    },
                    ensure_ascii=False,
                )
            )
    else:
        for rank, result in enumerate(results, start=1):
            preview = result.text.replace("\n", " ")[:100]
            print(f"{rank:>3}. {result.score:+.6f}  {result.id}\n     {preview}")
        if not results:
            print("no results")
    return 0


async def _cmd_stats(args: argparse.Namespace) -> int:
    embedder = _build_embedder(args)
    with SemanticIndex.open(Path(args.path), embedder) as index:
        print(json.dumps(index.stats(), indent=2))
    return 0


async def _cmd_compact(args: argparse.Namespace) -> int:
    embedder = _build_embedder(args)
    with SemanticIndex.open(Path(args.path), embedder) as index:
        before = index.size()
        dropped = index.compact()
    print(f"dropped {dropped} tombstoned rows ({before} -> {before - dropped})")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="simple-semantic", description="brute-force semantic search"
    )
    parser.add_argument("--embedder", default="hashing", choices=["hashing", "replay", "gemini"])
    parser.add_argument("--fixture", type=Path, help="recorded embeddings, for --embedder replay")
    parser.add_argument("--dimension", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0, help="hashing embedder seed")
    parser.add_argument("--model", default="gemini-embedding-001")

    sub = parser.add_subparsers(dest="command", required=True)

    index_cmd = sub.add_parser("index", help="add JSONL documents to an index")
    index_cmd.add_argument("path")
    index_cmd.add_argument("-i", "--input", type=Path, help="JSONL file (default: stdin)")
    index_cmd.add_argument("--chunk", action="store_true", help="chunk before indexing")
    index_cmd.add_argument("--chunk-size", type=int, default=512)
    index_cmd.add_argument("--chunk-overlap", type=int, default=64)
    index_cmd.set_defaults(run=_cmd_index)

    search_cmd = sub.add_parser("search", help="query an index")
    search_cmd.add_argument("path")
    search_cmd.add_argument("query")
    search_cmd.add_argument("-k", type=int, default=10)
    search_cmd.add_argument("--json", action="store_true")
    search_cmd.set_defaults(run=_cmd_search)

    stats_cmd = sub.add_parser("stats", help="print index statistics")
    stats_cmd.add_argument("path")
    stats_cmd.set_defaults(run=_cmd_stats)

    compact_cmd = sub.add_parser("compact", help="drop tombstoned rows")
    compact_cmd.add_argument("path")
    compact_cmd.set_defaults(run=_cmd_compact)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(asyncio.run(args.run(args)))
    except SimpleSemanticError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
