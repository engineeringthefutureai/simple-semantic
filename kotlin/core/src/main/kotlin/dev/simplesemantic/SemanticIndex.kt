package dev.simplesemantic

import java.io.RandomAccessFile
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption
import java.nio.file.StandardOpenOption
import java.util.PriorityQueue

/** One indexable unit. Already chunked — see [FixedChunker]. */
public data class Document(
    val id: String,
    val text: String,
    val meta: Map<String, Any?> = emptyMap(),
)

public data class SearchResult(
    val id: String,
    val score: Double,
    val text: String,
    val meta: Map<String, Any?>,
    val row: Int,
)

/**
 * What [SemanticIndex.addAll] did. [skipped] counts documents whose content
 * hash already matched a live row, so no embedding call was made.
 */
public data class AddResult(val added: Int, val replaced: Int, val skipped: Int)

public const val DEFAULT_CHUNKER_ID: String = "none"

/**
 * A directory of four files, plus the in-memory maps needed to serve it.
 *
 * Loaded at open: ids, metadata and content hashes. Not loaded: document text,
 * fetched through `offsets.bin` on demand.
 */
public class SemanticIndex private constructor(
    public val path: Path,
    public val embedder: Embedder,
) : AutoCloseable {

    private lateinit var current: Manifest
    private val ids = ArrayList<String>()
    private val hashes = ArrayList<String>()
    private val byId = HashMap<String, Int>()
    private var offsets: LongArray = longArrayOf(0)
    private var tombstones: Tombstones = Tombstones.empty(0)
    private var vectors: VectorStore? = null

    public val manifest: Manifest get() = current

    public companion object {
        /** Create an empty index directory. */
        public fun create(
            path: Path,
            embedder: Embedder,
            chunkerId: String = DEFAULT_CHUNKER_ID,
            existOk: Boolean = false,
        ): SemanticIndex {
            val manifestPath = path.resolve(FileNames.MANIFEST)
            if (Files.exists(manifestPath) && !existOk) {
                throw SimpleSemanticException(
                    "$path already contains a ${FileNames.MANIFEST}; " +
                        "pass existOk = true to overwrite it",
                )
            }
            Files.createDirectories(path)

            val now = utcNow()
            val manifest = Manifest(
                embedderId = embedder.id,
                dimension = embedder.dimension,
                chunkerId = chunkerId,
                rowCount = 0,
                liveCount = 0,
                createdAt = now,
                updatedAt = now,
            )
            Files.write(path.resolve(FileNames.VECTORS), ByteArray(0))
            Files.write(path.resolve(FileNames.DOCS), ByteArray(0))
            Files.write(path.resolve(FileNames.OFFSETS), encodeOffsets(longArrayOf(0)))
            writeAtomic(manifestPath, manifest.encode().toByteArray(Charsets.UTF_8))

            return open(path, embedder)
        }

        /** Open an existing index, running every check in SPEC.md §2.2. */
        public fun open(path: Path, embedder: Embedder): SemanticIndex {
            val manifestPath = path.resolve(FileNames.MANIFEST)
            if (!Files.exists(manifestPath)) {
                throw CorruptIndexException(
                    "$manifestPath: no ${FileNames.MANIFEST}; not a simple-semantic index",
                )
            }
            val manifest = Manifest.decode(
                Files.readString(manifestPath, Charsets.UTF_8),
                manifestPath.toString(),
            )

            // SPEC.md §2.1. Do not relax this into a warning.
            if (manifest.embedderId != embedder.id) {
                throw EmbedderMismatchException(path.toString(), manifest.embedderId, embedder.id)
            }
            if (manifest.dimension != embedder.dimension) {
                throw CorruptIndexException(
                    "$path: manifest dimension ${manifest.dimension} but embedder " +
                        "'${embedder.id}' produces ${embedder.dimension}",
                )
            }

            val index = SemanticIndex(path, embedder)
            index.current = manifest
            index.offsets = readOffsets(path, manifest.rowCount)
            index.tombstones = readTombstones(path, manifest.rowCount)
            index.vectors = VectorStore.open(
                path.resolve(FileNames.VECTORS),
                manifest.rowCount,
                manifest.dimension,
            )
            index.loadDocs()

            val live = manifest.rowCount - index.tombstones.deletedCount
            if (live != manifest.liveCount) {
                throw CorruptIndexException(
                    "$path: manifest live_count ${manifest.liveCount} disagrees with " +
                        "${FileNames.TOMBSTONES} ($live live of ${manifest.rowCount})",
                )
            }
            return index
        }

        private fun readOffsets(path: Path, rowCount: Int): LongArray {
            val file = path.resolve(FileNames.OFFSETS)
            val raw = Files.readAllBytes(file)
            val expected = (rowCount + 1) * 8
            if (raw.size != expected) {
                throw CorruptIndexException(
                    "$file is ${raw.size} bytes but row_count $rowCount requires exactly " +
                        "$expected ((row_count + 1) * 8)",
                )
            }
            return LongArray(rowCount + 1) { i ->
                var value = 0L
                for (b in 7 downTo 0) {
                    value = (value shl 8) or (raw[i * 8 + b].toLong() and 0xFF)
                }
                value
            }
        }

        private fun readTombstones(path: Path, rowCount: Int): Tombstones {
            val file = path.resolve(FileNames.TOMBSTONES)
            // An absent file means no deletions. SPEC.md §6.
            if (!Files.exists(file)) return Tombstones.empty(rowCount)
            return Tombstones.of(Files.readAllBytes(file), rowCount)
        }

        /** Unsigned 64-bit little-endian, per SPEC.md §5. */
        private fun encodeOffsets(values: LongArray): ByteArray {
            val out = ByteArray(values.size * 8)
            for (i in values.indices) {
                var value = values[i]
                for (b in 0 until 8) {
                    out[i * 8 + b] = (value and 0xFF).toByte()
                    value = value ushr 8
                }
            }
            return out
        }

        /** Write via a sibling temp file and rename; a reader never sees a partial file. */
        private fun writeAtomic(target: Path, data: ByteArray) {
            val temp = target.resolveSibling("${target.fileName}.tmp")
            Files.newOutputStream(
                temp,
                StandardOpenOption.CREATE,
                StandardOpenOption.WRITE,
                StandardOpenOption.TRUNCATE_EXISTING,
            ).use { stream ->
                stream.write(data)
                stream.flush()
            }
            syncFile(temp)
            Files.move(
                temp,
                target,
                StandardCopyOption.REPLACE_EXISTING,
                StandardCopyOption.ATOMIC_MOVE,
            )
            syncDirectory(target.parent)
        }

        private fun syncFile(file: Path) {
            java.nio.channels.FileChannel.open(file, StandardOpenOption.WRITE).use { it.force(true) }
        }

        private fun syncDirectory(directory: Path) {
            // Not portable: on POSIX this is what makes a rename durable,
            // elsewhere failing to open the directory is not worth propagating.
            runCatching {
                java.nio.channels.FileChannel.open(directory, StandardOpenOption.READ)
                    .use { it.force(true) }
            }
        }
    }

    /**
     * Scan docs.jsonl once, building the id/meta/hash arrays and the id map.
     *
     * Ascending order with "latest live row wins": the previous row for an id
     * was tombstoned before the new one was appended.
     */
    private fun loadDocs() {
        ids.clear()
        hashes.clear()
        byId.clear()
        val docsPath = path.resolve(FileNames.DOCS)
        Files.newBufferedReader(docsPath, Charsets.UTF_8).use { reader ->
            var line = reader.readLine()
            while (line != null) {
                val obj = CanonicalJson.parseObject(line)
                ids.add(obj["id"] as? String ?: throw CorruptIndexException("$docsPath: missing id"))
                hashes.add(obj["hash"] as? String ?: "")
                line = reader.readLine()
            }
        }
        if (ids.size != current.rowCount) {
            throw CorruptIndexException(
                "$docsPath has ${ids.size} lines but the manifest declares " +
                    "row_count ${current.rowCount}",
            )
        }
        for (row in ids.indices) {
            if (!tombstones.isDeleted(row)) byId[ids[row]] = row
        }
    }

    // ------------------------------------------------------------- inspection

    /** Total rows, live and tombstoned. */
    public fun size(): Int = current.rowCount

    public fun liveCount(): Int = current.liveCount

    public fun contains(id: String): Boolean = byId.containsKey(id)

    public fun get(id: String): Document? = byId[id]?.let { documentAt(it) }

    /** Live ids, in row order. */
    public fun ids(): List<String> = byId.values.sorted().map { ids[it] }

    public fun stats(): Map<String, Any?> = linkedMapOf(
        "path" to path.toString(),
        "embedder_id" to current.embedderId,
        "chunker_id" to current.chunkerId,
        "dimension" to current.dimension,
        "row_count" to current.rowCount,
        "live_count" to current.liveCount,
        "deleted_count" to (current.rowCount - current.liveCount),
        "vectors_bytes" to current.rowCount.toLong() * current.dimension * 4,
        "created_at" to current.createdAt,
        "updated_at" to current.updatedAt,
    )

    /** Fetch one document by row, via offsets.bin. SPEC.md §5. */
    private fun documentAt(row: Int): Document {
        val start = offsets[row]
        val end = offsets[row + 1]
        val buffer = ByteArray((end - start).toInt())
        RandomAccessFile(path.resolve(FileNames.DOCS).toFile(), "r").use { file ->
            file.seek(start)
            file.readFully(buffer)
        }
        val line = String(buffer, Charsets.UTF_8).trimEnd('\n')
        val obj = CanonicalJson.parseObject(line)
        @Suppress("UNCHECKED_CAST")
        return Document(
            id = obj["id"] as String,
            text = obj["text"] as String,
            meta = (obj["meta"] as? Map<String, Any?>) ?: emptyMap(),
        )
    }

    // ------------------------------------------------------------------ writes

    /**
     * Add or replace documents. Unchanged content is not re-embedded, keyed on
     * the content hash from SPEC.md §4.1.
     */
    public suspend fun addAll(documents: List<Document>): AddResult {
        if (documents.isEmpty()) return AddResult(0, 0, 0)

        val pending = ArrayList<Pair<Document, String>>()
        var skipped = 0
        // Track hashes staged in this call so a repeat inside one batch is also
        // a no-op rather than an append of an identical row.
        val staged = HashMap<String, String>()
        for (document in documents) {
            if (document.id.isEmpty()) {
                throw SimpleSemanticException("document id must be a non-empty string")
            }
            val digest = contentHash(document.text, embedder.id, current.chunkerId)
            val stagedHash = staged[document.id]
            if (stagedHash != null) {
                if (stagedHash == digest) {
                    skipped++
                    continue
                }
            } else {
                val existing = byId[document.id]
                if (existing != null && hashes[existing] == digest) {
                    skipped++
                    continue
                }
            }
            staged[document.id] = digest
            pending.add(document to digest)
        }

        if (pending.isEmpty()) return AddResult(0, 0, skipped)

        // Validate before embedding: an embedding call costs money.
        for ((document, _) in pending) CanonicalJson.validateMeta(document.meta)

        val embedded = embedInBatches(pending.map { it.first.text })

        var replaced = 0
        val vectorBytes = java.io.ByteArrayOutputStream()
        val docBytes = java.io.ByteArrayOutputStream()
        val newOffsets = ArrayList<Long>(pending.size)
        var nextOffset = offsets.last()
        val firstNewRow = current.rowCount
        // Grow the bitmap up front: an id repeated inside this same batch
        // tombstones a row that only exists because of an earlier iteration.
        tombstones.growTo(firstNewRow + pending.size)

        for (i in pending.indices) {
            val (document, digest) = pending[i]
            byId[document.id]?.let { previous ->
                tombstones.markDeleted(previous)
                replaced++
            }
            val row = firstNewRow + i
            // Normalized at write time whatever the embedder claims. SPEC.md §3.1.
            vectorBytes.write(normalizeRow(embedded[i]).toLittleEndianBytes())
            val line = CanonicalJson.encodeDocument(
                document.id,
                document.text,
                document.meta,
                digest,
            ) + "\n"
            val lineBytes = line.toByteArray(Charsets.UTF_8)
            docBytes.write(lineBytes)
            nextOffset += lineBytes.size
            newOffsets.add(nextOffset)
            ids.add(document.id)
            hashes.add(digest)
            byId[document.id] = row
        }

        append(vectorBytes.toByteArray(), docBytes.toByteArray(), newOffsets)
        return AddResult(added = pending.size - replaced, replaced = replaced, skipped = skipped)
    }

    /** Add one document, replacing any live row with the same id. */
    public suspend fun upsert(document: Document): AddResult = addAll(listOf(document))

    /**
     * Embed in batches the embedder declares it can take. Batching lives here so
     * no code path can call the embedder once per document over results.
     */
    private suspend fun embedInBatches(texts: List<String>): List<FloatArray> {
        val limit = maxOf(1, embedder.maxBatchSize)
        val out = ArrayList<FloatArray>(texts.size)
        var start = 0
        while (start < texts.size) {
            val batch = texts.subList(start, minOf(start + limit, texts.size))
            val vectors = embedder.embedDocuments(batch)
            if (vectors.size != batch.size) {
                throw SimpleSemanticException(
                    "embedder '${embedder.id}' returned ${vectors.size} vectors for " +
                        "${batch.size} texts",
                )
            }
            for (vector in vectors) {
                if (vector.size != current.dimension) {
                    throw SimpleSemanticException(
                        "embedder '${embedder.id}' returned ${vector.size} dimensions, " +
                            "expected ${current.dimension}",
                    )
                }
            }
            out.addAll(vectors)
            start += limit
        }
        return out
    }

    /**
     * Append to all files, then commit by rewriting the manifest.
     *
     * The manifest is written last because it declares how long the others
     * should be: a crash before it leaves a vectors.f32 longer than row_count
     * implies, which SPEC.md §2.2's length check catches on the next open.
     */
    private fun append(vectorBytes: ByteArray, docBytes: ByteArray, newOffsets: List<Long>) {
        closeMapping()
        Files.newOutputStream(
            path.resolve(FileNames.VECTORS),
            StandardOpenOption.WRITE,
            StandardOpenOption.APPEND,
        ).use { it.write(vectorBytes) }
        Files.newOutputStream(
            path.resolve(FileNames.DOCS),
            StandardOpenOption.WRITE,
            StandardOpenOption.APPEND,
        ).use { it.write(docBytes) }

        val combined = LongArray(offsets.size + newOffsets.size)
        offsets.copyInto(combined)
        for (i in newOffsets.indices) combined[offsets.size + i] = newOffsets[i]
        offsets = combined

        val rowCount = ids.size
        tombstones.growTo(rowCount)

        writeAtomic(path.resolve(FileNames.OFFSETS), encodeOffsets(offsets))
        writeTombstones()

        current = current.copy(
            rowCount = rowCount,
            liveCount = rowCount - tombstones.deletedCount,
            updatedAt = utcNow(),
        )
        writeAtomic(path.resolve(FileNames.MANIFEST), current.encode().toByteArray(Charsets.UTF_8))

        vectors = VectorStore.open(
            path.resolve(FileNames.VECTORS),
            rowCount,
            current.dimension,
        )
    }

    private fun writeTombstones() {
        val file = path.resolve(FileNames.TOMBSTONES)
        if (tombstones.anyDeleted()) {
            writeAtomic(file, tombstones.toByteArray())
        } else {
            Files.deleteIfExists(file)
        }
    }

    /** Drop the mapping before the underlying file changes length. */
    private fun closeMapping() {
        vectors?.close()
        vectors = null
    }

    /** Tombstone the live row for [id]. Unknown ids are a no-op, not an error. */
    public fun delete(id: String): Boolean {
        val row = byId.remove(id) ?: return false
        tombstones.markDeleted(row)
        writeTombstones()
        current = current.copy(
            liveCount = current.rowCount - tombstones.deletedCount,
            updatedAt = utcNow(),
        )
        writeAtomic(path.resolve(FileNames.MANIFEST), current.encode().toByteArray(Charsets.UTF_8))
        return true
    }

    /**
     * Rewrite the index without tombstoned rows. Returns rows dropped.
     *
     * Crash-safe: the new index is built complete in a sibling directory and
     * synced before anything in place is touched.
     */
    public fun compact(): Int {
        val dropped = current.rowCount - current.liveCount
        if (dropped == 0) return 0

        val liveRows = (0 until current.rowCount).filter { !tombstones.isDeleted(it) }
        val staging = path.resolveSibling("${path.fileName}.compact.tmp")
        if (Files.exists(staging)) deleteRecursively(staging)
        Files.createDirectories(staging)

        val store = vectors ?: throw SimpleSemanticException("index is closed")
        val newOffsets = ArrayList<Long>(liveRows.size + 1)
        newOffsets.add(0)
        var cursor = 0L

        Files.newOutputStream(staging.resolve(FileNames.VECTORS), StandardOpenOption.CREATE_NEW)
            .use { vectorOut ->
                Files.newOutputStream(staging.resolve(FileNames.DOCS), StandardOpenOption.CREATE_NEW)
                    .use { docOut ->
                        RandomAccessFile(path.resolve(FileNames.DOCS).toFile(), "r").use { docIn ->
                            for (row in liveRows) {
                                vectorOut.write(store.row(row).toLittleEndianBytes())
                                val start = offsets[row]
                                val length = (offsets[row + 1] - start).toInt()
                                val buffer = ByteArray(length)
                                docIn.seek(start)
                                docIn.readFully(buffer)
                                docOut.write(buffer)
                                cursor += length
                                newOffsets.add(cursor)
                            }
                        }
                    }
            }

        writeAtomic(staging.resolve(FileNames.OFFSETS), encodeOffsets(newOffsets.toLongArray()))
        val compacted = current.copy(
            rowCount = liveRows.size,
            liveCount = liveRows.size,
            updatedAt = utcNow(),
        )
        writeAtomic(
            staging.resolve(FileNames.MANIFEST),
            compacted.encode().toByteArray(Charsets.UTF_8),
        )

        closeMapping()
        val retired = path.resolveSibling("${path.fileName}.compact.old")
        if (Files.exists(retired)) deleteRecursively(retired)
        Files.move(path, retired, StandardCopyOption.ATOMIC_MOVE)
        Files.move(staging, path, StandardCopyOption.ATOMIC_MOVE)
        deleteRecursively(retired)

        current = Manifest.decode(
            Files.readString(path.resolve(FileNames.MANIFEST), Charsets.UTF_8),
            path.resolve(FileNames.MANIFEST).toString(),
        )
        offsets = readOffsets(path, current.rowCount)
        tombstones = readTombstones(path, current.rowCount)
        vectors = VectorStore.open(
            path.resolve(FileNames.VECTORS),
            current.rowCount,
            current.dimension,
        )
        loadDocs()
        return dropped
    }

    private fun deleteRecursively(directory: Path) {
        Files.walk(directory).sorted(Comparator.reverseOrder()).forEach { Files.deleteIfExists(it) }
    }

    // ------------------------------------------------------------------ search

    /** Embed the query, then run exact k-NN over the live rows. */
    public suspend fun search(query: String, k: Int = 10): List<SearchResult> {
        // No direction; better than ranking against the e_0 fallback.
        if (query.isBlank()) return emptyList()
        return searchVector(embedder.embedQuery(query), k)
    }

    /**
     * Exact k-NN against a pre-computed query vector. Public so a caller with a
     * cached embedding or a centroid need not go back through the embedder.
     */
    public fun searchVector(queryVector: FloatArray, k: Int = 10): List<SearchResult> {
        if (k <= 0 || current.rowCount == 0) return emptyList()
        if (queryVector.size != current.dimension) {
            throw SimpleSemanticException(
                "query vector has ${queryVector.size} dimensions, expected ${current.dimension}",
            )
        }
        val store = vectors ?: throw SimpleSemanticException("index is closed")
        val query = normalizeRow(queryVector)

        val live = tombstones.liveMask()

        // A bounded min-heap: O(n log k), never a full sort. The head is the
        // worst candidate held, so admitting a new row is one comparison.
        val heap = PriorityQueue<Hit>(k, WORST_FIRST)
        for (row in 0 until current.rowCount) {
            if (!live[row]) continue
            val score = store.dot(row, query)
            if (heap.size < k) {
                heap.add(Hit(row, score))
            } else {
                val worst = heap.peek()
                // Ties break by ascending row index. SPEC.md §8.
                if (score > worst.score || (score == worst.score && row < worst.row)) {
                    heap.poll()
                    heap.add(Hit(row, score))
                }
            }
        }

        return heap.sortedWith(BEST_FIRST).map { hit ->
            val document = documentAt(hit.row)
            SearchResult(
                id = document.id,
                score = hit.score,
                text = document.text,
                meta = document.meta,
                row = hit.row,
            )
        }
    }

    override fun close() {
        closeMapping()
    }
}

private data class Hit(val row: Int, val score: Double)

/** Worst candidate at the head: lowest score, and among equals the highest row. */
private val WORST_FIRST: Comparator<Hit> =
    compareBy<Hit> { it.score }.thenByDescending { it.row }

/** Final ordering: highest score first, ties by ascending row. SPEC.md §8. */
private val BEST_FIRST: Comparator<Hit> =
    compareByDescending<Hit> { it.score }.thenBy { it.row }
