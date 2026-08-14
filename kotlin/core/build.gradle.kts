plugins {
    kotlin("plugin.serialization")
}

dependencies {
    // SPEC.md §7 pins a canonical *encoding* that no library guarantees by
    // default, so the encoder is hand-written. Decoding has no such constraint,
    // so it is declarative: @Serializable data classes rather than casts out of
    // a Map<String, Any?>.
    //
    // No BLAS: the hand-written dot loop is the point.
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.9.0")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.7.3")
}
