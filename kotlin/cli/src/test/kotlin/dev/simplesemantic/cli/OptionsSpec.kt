package dev.simplesemantic.cli

import io.kotest.assertions.throwables.shouldThrow
import io.kotest.core.spec.style.StringSpec
import io.kotest.matchers.collections.shouldContainExactly
import io.kotest.matchers.shouldBe
import io.kotest.matchers.string.shouldContain

/** Argument parsing. The commands themselves are covered by conformance/run.sh. */
class OptionsSpec : StringSpec({

    "a query is the one argument after the path" {
        val options = Options.parse(arrayOf("search", "./idx", "some query", "-k", "3"))!!
        options.command shouldBe "search"
        options.path.toString() shouldBe "./idx"
        options.arguments shouldContainExactly listOf("some query")
        options.k shouldBe 3
    }

    "delete takes any number of ids" {
        val options = Options.parse(arrayOf("delete", "./idx", "a", "b", "c"))!!
        options.arguments shouldContainExactly listOf("a", "b", "c")
    }

    "a flag with no value is an error, not an index out of bounds" {
        val error = shouldThrow<IllegalArgumentException> {
            Options.parse(arrayOf("search", "./idx", "query", "--dimension"))
        }
        error.message!! shouldContain "--dimension expects a value"
    }

    "too few positional arguments asks for usage" {
        Options.parse(arrayOf("stats")) shouldBe null
        Options.parse(arrayOf()) shouldBe null
        Options.parse(arrayOf("--help")) shouldBe null
    }
})
