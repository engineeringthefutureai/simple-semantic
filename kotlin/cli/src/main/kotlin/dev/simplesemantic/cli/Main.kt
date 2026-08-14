package dev.simplesemantic.cli

import dev.simplesemantic.CanonicalJson
import dev.simplesemantic.DEFAULT_CHUNKER_ID
import dev.simplesemantic.Document
import dev.simplesemantic.Embedder
import dev.simplesemantic.FileNames
import dev.simplesemantic.FixedChunker
import dev.simplesemantic.GeminiEmbedder
import dev.simplesemantic.HashingEmbedder
import dev.simplesemantic.SemanticIndex
import dev.simplesemantic.SimpleSemanticException
import java.nio.file.Files
import java.nio.file.Path
import kotlin.system.exitProcess
import kotlinx.coroutines.runBlocking

/**
 * `simple-semantic` command line: index, search, stats, compact.
 *
 * Deliberately thin. It exists so that an index can be inspected and driven
 * without writing a program, and so the conformance harness has something to
 * call.
 */
public fun main(args: Array<String>) {
    try {
        exitProcess(run(args))
    } catch (exc: SimpleSemanticException) {
        System.err.println("error: ${exc.message}")
        exitProcess(1)
    } catch (exc: IllegalArgumentException) {
        System.err.println("error: ${exc.message}")
        exitProcess(2)
    }
}

private fun run(args: Array<String>): Int {
    val options = Options.parse(args) ?: run {
        printUsage()
        return 2
    }
    return when (options.command) {
        "index" -> commandIndex(options)
        "search" -> commandSearch(options)
        "stats" -> commandStats(options)
        "compact" -> commandCompact(options)
        else -> {
            printUsage()
            2
        }
    }
}

private fun buildEmbedder(options: Options): Embedder = when (options.embedder) {
    "hashing" -> HashingEmbedder(dimension = options.dimension, seed = options.seed)
    "gemini" -> GeminiEmbedder(model = options.model, dimension = options.dimension)
    else -> throw SimpleSemanticException("unknown embedder '${options.embedder}'")
}

/**
 * Read JSONL documents from a file or stdin.
 *
 * Each line is `{"id": ..., "text": ..., "meta": {...}}` — the same shape as
 * docs.jsonl minus the hash, which the index computes.
 */
private fun readDocuments(source: Path?): List<Document> {
    val lines = if (source == null) {
        generateSequence(::readLine).toList()
    } else {
        Files.readAllLines(source, Charsets.UTF_8)
    }
    return lines.mapIndexedNotNull { number, raw ->
        val line = raw.trim()
        if (line.isEmpty()) return@mapIndexedNotNull null
        val obj = try {
            CanonicalJson.parseObject(line)
        } catch (exc: SimpleSemanticException) {
            throw SimpleSemanticException("line ${number + 1}: not valid JSON (${exc.message})")
        }
        val id = obj["id"] as? String
        val text = obj["text"] as? String
        if (id == null || text == null) {
            throw SimpleSemanticException("line ${number + 1}: needs both 'id' and 'text'")
        }
        @Suppress("UNCHECKED_CAST")
        Document(id, text, (obj["meta"] as? Map<String, Any?>) ?: emptyMap())
    }
}

private fun commandIndex(options: Options): Int = runBlocking {
    val embedder = buildEmbedder(options)
    var documents = readDocuments(options.input)

    val chunkerId = if (options.chunk) {
        val chunker = FixedChunker(options.chunkSize, options.chunkOverlap)
        documents = documents.flatMap { document ->
            chunker.chunk(document.text).map { piece ->
                Document("${document.id}#${piece.index}", piece.text, document.meta)
            }
        }
        chunker.id
    } else {
        DEFAULT_CHUNKER_ID
    }

    val path = options.path
    val index = if (Files.exists(path.resolve(FileNames.MANIFEST))) {
        SemanticIndex.open(path, embedder)
    } else {
        SemanticIndex.create(path, embedder, chunkerId)
    }

    index.use {
        val result = it.addAll(documents)
        println(
            "added ${result.added}, replaced ${result.replaced}, " +
                "skipped ${result.skipped} (unchanged) -> ${it.liveCount()} live rows",
        )
    }
    0
}

private fun commandSearch(options: Options): Int = runBlocking {
    val embedder = buildEmbedder(options)
    SemanticIndex.open(options.path, embedder).use { index ->
        val filter = options.filters.takeIf { it.isNotEmpty() }?.let { wanted ->
            { meta: Map<String, Any?> -> wanted.all { (key, value) -> meta[key]?.toString() == value } }
        }
        val results = index.search(options.query.orEmpty(), options.k, filter)

        if (options.json) {
            for (result in results) {
                println(
                    CanonicalJson.encodeObject(
                        linkedMapOf(
                            "id" to result.id,
                            // Emitted as a string: SPEC.md §7.2 keeps floats
                            // out of canonical JSON, and the shortest decimal
                            // form of a double is not portable anyway.
                            "score" to result.score.toString(),
                            "row" to result.row,
                            "text" to result.text,
                            "meta" to result.meta,
                        ),
                    ),
                )
            }
        } else {
            if (results.isEmpty()) println("no results")
            results.forEachIndexed { rank, result ->
                val preview = result.text.replace("\n", " ").take(100)
                println(String.format("%3d. %+.6f  %s", rank + 1, result.score, result.id))
                println("     $preview")
            }
        }
    }
    0
}

private fun commandStats(options: Options): Int {
    SemanticIndex.open(options.path, buildEmbedder(options)).use { index ->
        for ((key, value) in index.stats()) {
            println("$key: $value")
        }
    }
    return 0
}

private fun commandCompact(options: Options): Int {
    SemanticIndex.open(options.path, buildEmbedder(options)).use { index ->
        val before = index.size()
        val dropped = index.compact()
        println("dropped $dropped tombstoned rows ($before -> ${before - dropped})")
    }
    return 0
}

private class Options(
    val command: String,
    val path: Path,
    val query: String?,
    val embedder: String,
    val dimension: Int,
    val seed: Long,
    val model: String,
    val k: Int,
    val filters: Map<String, String>,
    val json: Boolean,
    val input: Path?,
    val chunk: Boolean,
    val chunkSize: Int,
    val chunkOverlap: Int,
) {
    companion object {
        @Suppress("CyclomaticComplexMethod")
        fun parse(args: Array<String>): Options? {
            if (args.isEmpty()) return null
            var command: String? = null
            var path: Path? = null
            var query: String? = null
            var embedder = "hashing"
            var dimension = 256
            var seed = 0L
            var model = "gemini-embedding-001"
            var k = 10
            val filters = LinkedHashMap<String, String>()
            var json = false
            var input: Path? = null
            var chunk = false
            var chunkSize = 512
            var chunkOverlap = 64

            var i = 0
            while (i < args.size) {
                when (val arg = args[i]) {
                    "--embedder" -> embedder = args[++i]
                    "--dimension" -> dimension = args[++i].toInt()
                    "--seed" -> seed = args[++i].toLong()
                    "--model" -> model = args[++i]
                    "-k" -> k = args[++i].toInt()
                    "--filter" -> {
                        val pair = args[++i].split("=", limit = 2)
                        require(pair.size == 2) { "--filter expects KEY=VALUE, got '${args[i]}'" }
                        filters[pair[0]] = pair[1]
                    }
                    "--json" -> json = true
                    "-i", "--input" -> input = Path.of(args[++i])
                    "--chunk" -> chunk = true
                    "--chunk-size" -> chunkSize = args[++i].toInt()
                    "--chunk-overlap" -> chunkOverlap = args[++i].toInt()
                    "-h", "--help" -> return null
                    else -> when {
                        command == null -> command = arg
                        path == null -> path = Path.of(arg)
                        query == null -> query = arg
                        else -> throw IllegalArgumentException("unexpected argument '$arg'")
                    }
                }
                i++
            }

            if (command == null || path == null) return null
            return Options(
                command, path, query, embedder, dimension, seed, model, k,
                filters, json, input, chunk, chunkSize, chunkOverlap,
            )
        }
    }
}

private fun printUsage() {
    System.err.println(
        """
        simple-semantic — brute-force semantic search

        usage:
          simple-semantic index   <path> [-i FILE] [--chunk] [--chunk-size N] [--chunk-overlap N]
          simple-semantic search  <path> <query> [-k N] [--filter KEY=VALUE]... [--json]
          simple-semantic stats   <path>
          simple-semantic compact <path>

        embedder options (must match the index — see SPEC.md §2.1):
          --embedder hashing|gemini   default: hashing
          --dimension N               default: 256
          --seed N                    hashing embedder seed, default: 0
          --model NAME                default: gemini-embedding-001

        'index' reads JSONL from stdin unless -i is given:
          {"id": "doc1", "text": "...", "meta": {"source": "notes"}}
        """.trimIndent(),
    )
}
