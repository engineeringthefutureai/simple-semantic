dependencies {
    // core has no runtime dependencies beyond the Kotlin stdlib and coroutines.
    // No JSON library: SPEC.md §7 pins an encoding no library guarantees by
    // default, so the encoder is hand-written. No BLAS: the hand-written dot
    // loop is the point.
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.9.0")
}
