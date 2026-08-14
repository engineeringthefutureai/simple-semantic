package dev.simplesemantic

/**
 * The canonical JSON encoder — SPEC.md §7.
 *
 * Hand-written because no JVM JSON library guarantees all four rules by default:
 * separator style, key ordering, escape selection, and literal non-ASCII output,
 * and byte-identity between the two implementations depends on all four.
 *
 * Decoding has no such constraint and is declarative — see Wire.kt.
 */
public object CanonicalJson {

    /**
     * Encode an object whose key order is already correct. Pass a
     * [LinkedHashMap] and insertion order is output order.
     */
    public fun encodeObject(entries: Map<String, Any?>, sortKeys: Boolean = false): String {
        val builder = StringBuilder()
        writeObject(builder, entries, sortKeys, "")
        return builder.toString()
    }

    /** One line of docs.jsonl, without the terminating LF. SPEC.md §4. */
    public fun encodeDocument(
        id: String,
        text: String,
        meta: Map<String, Any?>,
        hash: String,
    ): String {
        validateMeta(meta)
        val builder = StringBuilder()
        builder.append("{\"id\":")
        writeString(builder, id)
        builder.append(",\"text\":")
        writeString(builder, text)
        builder.append(",\"meta\":")
        // meta is user-supplied, so its keys sort; the four outer keys do not.
        writeObject(builder, meta, sortKeys = true, path = "meta")
        builder.append(",\"hash\":")
        writeString(builder, hash)
        builder.append('}')
        return builder.toString()
    }

    /** A single JSON string literal, escaped per §7 rule 3. */
    public fun encodeString(value: String): String {
        val builder = StringBuilder()
        writeString(builder, value)
        return builder.toString()
    }

    /** Reject anything the other implementation could not reproduce byte-for-byte. */
    public fun validateMeta(meta: Map<String, Any?>) {
        for ((key, value) in meta) {
            validateValue(value, "meta.$key")
        }
    }

    private fun validateValue(value: Any?, path: String) {
        when (value) {
            null, is String, is Boolean -> return
            is Byte, is Short, is Int, is Long -> return
            is List<*> -> value.forEachIndexed { i, item -> validateValue(item, "$path[$i]") }
            is Map<*, *> -> for ((key, item) in value) {
                if (key !is String) throw MetaValueException(path, key)
                validateValue(item, "$path.$key")
            }
            else -> throw MetaValueException(path, value)
        }
    }

    private fun writeValue(builder: StringBuilder, value: Any?, path: String) {
        when (value) {
            null -> builder.append("null")
            is String -> writeString(builder, value)
            is Boolean -> builder.append(if (value) "true" else "false")
            is Byte, is Short, is Int, is Long -> builder.append(value.toString())
            is List<*> -> {
                builder.append('[')
                value.forEachIndexed { i, item ->
                    if (i > 0) builder.append(',')
                    writeValue(builder, item, "$path[$i]")
                }
                builder.append(']')
            }
            is Map<*, *> -> {
                @Suppress("UNCHECKED_CAST")
                writeObject(builder, value as Map<String, Any?>, true, path)
            }
            else -> throw MetaValueException(path, value)
        }
    }

    private fun writeObject(
        builder: StringBuilder,
        entries: Map<String, Any?>,
        sortKeys: Boolean,
        path: String,
    ) {
        val keys = if (sortKeys) entries.keys.sortedWith(CODE_POINT_ORDER) else entries.keys.toList()
        builder.append('{')
        keys.forEachIndexed { i, key ->
            if (i > 0) builder.append(',')
            writeString(builder, key)
            builder.append(':')
            writeValue(builder, entries[key], if (path.isEmpty()) key else "$path.$key")
        }
        builder.append('}')
    }

    /**
     * Ascending by Unicode code point, per SPEC.md §7 rule 2.
     *
     * Not [String.compareTo], which orders by UTF-16 code unit and so sorts
     * astral-plane characters below U+E000..U+FFFF. Comparing UTF-8 bytes is
     * equivalent to comparing code points.
     */
    private val CODE_POINT_ORDER: Comparator<String> = Comparator { left, right ->
        val a = left.toByteArray(Charsets.UTF_8)
        val b = right.toByteArray(Charsets.UTF_8)
        var i = 0
        while (i < a.size && i < b.size) {
            val diff = (a[i].toInt() and 0xFF) - (b[i].toInt() and 0xFF)
            if (diff != 0) return@Comparator diff
            i++
        }
        a.size - b.size
    }

    private fun writeString(builder: StringBuilder, value: String) {
        builder.append('"')
        for (ch in value) {
            when {
                ch == '"' -> builder.append("\\\"")
                ch == '\\' -> builder.append("\\\\")
                ch == '\b' -> builder.append("\\b")
                ch == '\u000C' -> builder.append("\\f")
                ch == '\n' -> builder.append("\\n")
                ch == '\r' -> builder.append("\\r")
                ch == '\t' -> builder.append("\\t")
                ch < ' ' -> builder.append("\\u").append(HEX[ch.code])
                // Everything else literal, including all non-ASCII. '/' is never escaped.
                else -> builder.append(ch)
            }
        }
        builder.append('"')
    }

    /** Lowercase hex, per SPEC.md §7 rule 3. Only the C0 controls need it. */
    private val HEX: Array<String> = Array(0x20) { String.format("%04x", it) }
}
