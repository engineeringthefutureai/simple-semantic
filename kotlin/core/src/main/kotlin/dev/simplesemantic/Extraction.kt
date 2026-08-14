package dev.simplesemantic

import java.lang.reflect.Field

/**
 * Marks the field holding the document id.
 *
 * Targeted at [AnnotationTarget.FIELD] so that the idiomatic
 * `@SemanticId val id: String` lands where plain Java reflection can see it —
 * no `kotlin-reflect` dependency, no `@field:` prefix in user code.
 */
@Target(AnnotationTarget.FIELD)
@Retention(AnnotationRetention.RUNTIME)
public annotation class SemanticId

/**
 * Marks a field whose text is embedded.
 *
 * [order] pins concatenation: `Class.getDeclaredFields()` is not specified to
 * return declaration order, and reordering a document's parts changes its
 * vector.
 */
@Target(AnnotationTarget.FIELD)
@Retention(AnnotationRetention.RUNTIME)
public annotation class SemanticIndexed(val order: Int = 0)

/** Marks a field stored as filterable metadata but not embedded. */
@Target(AnnotationTarget.FIELD)
@Retention(AnnotationRetention.RUNTIME)
public annotation class SemanticMeta

/**
 * Build documents from three plain lambdas. No annotations, no reflection, and
 * no requirement that the input be a data class.
 */
public fun <T> documentsFrom(
    items: Iterable<T>,
    idOf: (T) -> String,
    textOf: (T) -> String,
    metaOf: ((T) -> Map<String, Any?>)? = null,
): List<Document> = items.map { item ->
    Document(id = idOf(item), text = textOf(item), meta = metaOf?.invoke(item) ?: emptyMap())
}

/**
 * Build documents from annotated fields. Nullable annotated fields are skipped
 * rather than throwing.
 */
public fun <T : Any> documentsFrom(
    items: Iterable<T>,
    separator: String = "\n\n",
): List<Document> {
    val list = items.toList()
    if (list.isEmpty()) return emptyList()

    val type = list.first().javaClass
    val plan = ExtractionPlan.of(type)
    return list.map { item -> plan.extract(item, separator) }
}

/** The resolved annotation layout for one class. Computed once per call. */
internal class ExtractionPlan private constructor(
    private val idField: Field,
    private val textFields: List<Field>,
    private val metaFields: List<Field>,
    private val type: Class<*>,
) {
    companion object {
        fun of(type: Class<*>): ExtractionPlan {
            var idField: Field? = null
            val indexed = ArrayList<Pair<Int, Field>>()
            val meta = ArrayList<Field>()

            for ((position, field) in type.declaredFields.withIndex()) {
                if (field.isSynthetic) continue
                field.isAccessible = true
                if (field.isAnnotationPresent(SemanticId::class.java)) {
                    if (idField != null) {
                        throw ExtractionException(
                            "${type.simpleName} annotates both '${idField!!.name}' and " +
                                "'${field.name}' with @SemanticId; exactly one field may be the id",
                        )
                    }
                    idField = field
                }
                field.getAnnotation(SemanticIndexed::class.java)?.let { annotation ->
                    // Declaration order is only the tiebreak; order = n pins it.
                    indexed.add((annotation.order * 1000 + position) to field)
                }
                if (field.isAnnotationPresent(SemanticMeta::class.java)) meta.add(field)
            }

            val resolvedId = idField ?: throw ExtractionException(
                "${type.simpleName} has no field annotated with @SemanticId. Annotate the " +
                    "id field, or use documentsFrom(items, idOf = ..., textOf = ...).",
            )
            if (indexed.isEmpty()) {
                throw ExtractionException(
                    "${type.simpleName} has no field annotated with @SemanticIndexed, so " +
                        "there is nothing to embed. Annotate at least one text field.",
                )
            }
            return ExtractionPlan(
                resolvedId,
                indexed.sortedBy { it.first }.map { it.second },
                meta,
                type,
            )
        }
    }

    fun extract(item: Any, separator: String): Document {
        val rawId = idField.get(item)
            ?: throw ExtractionException(
                "${type.simpleName}.${idField.name} is annotated @SemanticId but is null; " +
                    "a document id is required",
            )
        val text = textFields.mapNotNull { it.get(item)?.toString() }.joinToString(separator)
        val meta = LinkedHashMap<String, Any?>()
        for (field in metaFields) {
            // Absent rather than null.
            field.get(item)?.let { meta[field.name] = it }
        }
        return Document(id = rawId.toString(), text = text, meta = meta)
    }
}
