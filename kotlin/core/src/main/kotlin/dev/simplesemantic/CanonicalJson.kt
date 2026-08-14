package dev.simplesemantic

/**
 * Canonical JSON — SPEC.md §7. Hand-written, both directions.
 *
 * No JVM JSON library guarantees all four canonicalization rules by default:
 * separator style, key ordering, escape selection, and literal (unescaped)
 * non-ASCII output. Since the format's whole claim is that two implementations
 * emit identical bytes, an encoder that is "close enough by default" is not
 * usable — and a dependency that might change its defaults in a minor release
 * is worse than 150 lines that cannot.
 */
public object CanonicalJson {

    // ------------------------------------------------------------------ encode

    /**
     * Encode an object whose key order is already correct.
     *
     * Used for the manifest and for document lines, where SPEC.md fixes the
     * order. Pass a [LinkedHashMap] and the insertion order is the output
     * order.
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

    /** Reject anything the Python implementation could not reproduce byte-for-byte. */
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
     * Not [String.compareTo], which orders by UTF-16 code unit and therefore
     * sorts astral-plane characters (U+10000 and above, encoded as surrogate
     * pairs starting at 0xD800) *below* the BMP characters U+E000..U+FFFF.
     * Python sorts strings by code point, so the JVM's natural order would
     * disagree on exactly the inputs an emoji key produces.
     *
     * Comparing UTF-8 bytes is equivalent to comparing code points — that is a
     * design property of UTF-8 — and is cheaper than decoding both strings.
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
                // Everything else literal, including all non-ASCII. Escaping it
                // would still be valid JSON and would still break byte-identity.
                // '/' is never escaped.
                else -> builder.append(ch)
            }
        }
        builder.append('"')
    }

    /** Lowercase hex, per SPEC.md §7 rule 3. Only the C0 controls need it. */
    private val HEX: Array<String> = Array(0x20) { String.format("%04x", it) }

    // ------------------------------------------------------------------ decode

    /**
     * Parse a JSON object into `Map<String, Any?>`.
     *
     * Values come back as `String`, `Long`, `Double`, `Boolean`, `null`,
     * `List<Any?>` or `Map<String, Any?>`. Doubles are parsed but cannot be
     * re-encoded — see [MetaValueException] — so a foreign file with a float in
     * meta is readable and reports the problem only if you try to write it back.
     */
    public fun parseObject(input: String): Map<String, Any?> {
        val parser = Parser(input)
        parser.skipWhitespace()
        val value = parser.parseValue()
        parser.skipWhitespace()
        if (!parser.atEnd()) throw JsonException("trailing content at offset ${parser.offset}")
        @Suppress("UNCHECKED_CAST")
        return value as? Map<String, Any?>
            ?: throw JsonException("expected a JSON object, got ${value?.javaClass?.simpleName}")
    }

    private class Parser(private val input: String) {
        var offset: Int = 0

        fun atEnd(): Boolean = offset >= input.length

        fun skipWhitespace() {
            while (offset < input.length && input[offset].isJsonWhitespace()) offset++
        }

        private fun Char.isJsonWhitespace(): Boolean =
            this == ' ' || this == '\t' || this == '\n' || this == '\r'

        fun parseValue(): Any? {
            if (atEnd()) throw JsonException("unexpected end of input")
            return when (val ch = input[offset]) {
                '{' -> parseObject()
                '[' -> parseArray()
                '"' -> parseString()
                't' -> literal("true", true)
                'f' -> literal("false", false)
                'n' -> literal("null", null)
                else ->
                    if (ch == '-' || ch in '0'..'9') {
                        parseNumber()
                    } else {
                        throw JsonException("unexpected character '$ch' at offset $offset")
                    }
            }
        }

        private fun literal(text: String, value: Any?): Any? {
            if (!input.startsWith(text, offset)) {
                throw JsonException("expected '$text' at offset $offset")
            }
            offset += text.length
            return value
        }

        private fun parseObject(): Map<String, Any?> {
            expect('{')
            val entries = LinkedHashMap<String, Any?>()
            skipWhitespace()
            if (peek() == '}') {
                offset++
                return entries
            }
            while (true) {
                skipWhitespace()
                val key = parseString()
                skipWhitespace()
                expect(':')
                skipWhitespace()
                entries[key] = parseValue()
                skipWhitespace()
                when (val ch = next()) {
                    ',' -> continue
                    '}' -> return entries
                    else -> throw JsonException("expected ',' or '}' but found '$ch' at $offset")
                }
            }
        }

        private fun parseArray(): List<Any?> {
            expect('[')
            val items = ArrayList<Any?>()
            skipWhitespace()
            if (peek() == ']') {
                offset++
                return items
            }
            while (true) {
                skipWhitespace()
                items.add(parseValue())
                skipWhitespace()
                when (val ch = next()) {
                    ',' -> continue
                    ']' -> return items
                    else -> throw JsonException("expected ',' or ']' but found '$ch' at $offset")
                }
            }
        }

        private fun parseString(): String {
            expect('"')
            val builder = StringBuilder()
            while (true) {
                if (atEnd()) throw JsonException("unterminated string")
                when (val ch = input[offset++]) {
                    '"' -> return builder.toString()
                    '\\' -> builder.append(parseEscape())
                    else -> builder.append(ch)
                }
            }
        }

        private fun parseEscape(): Char {
            if (atEnd()) throw JsonException("unterminated escape")
            return when (val ch = input[offset++]) {
                '"', '\\', '/' -> ch
                'b' -> '\b'
                'f' -> '\u000C'
                'n' -> '\n'
                'r' -> '\r'
                't' -> '\t'
                'u' -> {
                    if (offset + 4 > input.length) throw JsonException("truncated \\u escape")
                    val hex = input.substring(offset, offset + 4)
                    offset += 4
                    hex.toIntOrNull(16)?.toChar()
                        ?: throw JsonException("invalid \\u escape '$hex'")
                }
                else -> throw JsonException("invalid escape '\\$ch' at offset $offset")
            }
        }

        private fun parseNumber(): Any {
            val start = offset
            if (peek() == '-') offset++
            while (!atEnd() && input[offset] in '0'..'9') offset++
            var integral = true
            if (!atEnd() && input[offset] == '.') {
                integral = false
                offset++
                while (!atEnd() && input[offset] in '0'..'9') offset++
            }
            if (!atEnd() && (input[offset] == 'e' || input[offset] == 'E')) {
                integral = false
                offset++
                if (!atEnd() && (input[offset] == '+' || input[offset] == '-')) offset++
                while (!atEnd() && input[offset] in '0'..'9') offset++
            }
            val text = input.substring(start, offset)
            return if (integral) {
                text.toLongOrNull() ?: throw JsonException("integer out of 64-bit range: $text")
            } else {
                text.toDouble()
            }
        }

        private fun peek(): Char? = if (atEnd()) null else input[offset]

        private fun next(): Char {
            if (atEnd()) throw JsonException("unexpected end of input")
            return input[offset++]
        }

        private fun expect(ch: Char) {
            if (atEnd() || input[offset] != ch) {
                throw JsonException("expected '$ch' at offset $offset")
            }
            offset++
        }
    }
}
