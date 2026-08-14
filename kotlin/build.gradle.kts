import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import org.jetbrains.kotlin.gradle.dsl.KotlinJvmProjectExtension

plugins {
    kotlin("jvm") version "2.2.20" apply false
}

allprojects {
    group = "dev.simplesemantic"
    version = "0.1.0"

    repositories {
        mavenCentral()
    }
}

// JDK 22 is the floor: the Foreign Function & Memory API is finalized there, so
// java.lang.foreign needs no --enable-preview and no --add-modules. SPEC.md §3
// explains why this project needs FFM rather than ByteBuffer — the 2 GB
// ByteBuffer cap is about 175k rows at 3072 dimensions, a ceiling a real
// knowledge base reaches.
//
// Checked here rather than left to a confusing "cannot find symbol
// java.lang.foreign.Arena" from the compiler.
val minimumJdk = JavaVersion.VERSION_22
require(JavaVersion.current() >= minimumJdk) {
    "simple-semantic needs JDK ${minimumJdk.majorVersion} or newer for the Foreign " +
        "Function & Memory API, but Gradle is running on ${JavaVersion.current()}. " +
        "Set JAVA_HOME to a JDK ${minimumJdk.majorVersion}+ installation."
}

subprojects {
    apply(plugin = "org.jetbrains.kotlin.jvm")

    extensions.configure<KotlinJvmProjectExtension>("kotlin") {
        compilerOptions {
            // Bytecode targets 22 even when built on a newer JDK, so the
            // artifact runs anywhere FFM is final rather than only on the
            // build machine's JDK.
            jvmTarget.set(JvmTarget.JVM_22)
            // The public API of a library should be explicit: every declaration
            // states its visibility and its return type, so a widening of the
            // surface is a visible diff rather than an accident.
            extraWarnings.set(true)
            allWarningsAsErrors.set(true)
        }
        explicitApi()
    }

    extensions.configure<JavaPluginExtension>("java") {
        sourceCompatibility = minimumJdk
        targetCompatibility = minimumJdk
    }

    dependencies {
        add("testImplementation", "io.kotest:kotest-runner-junit5:5.9.1")
        add("testImplementation", "io.kotest:kotest-assertions-core:5.9.1")
        add("testImplementation", "org.jetbrains.kotlinx:kotlinx-coroutines-test:1.9.0")
    }

    tasks.withType<Test>().configureEach {
        useJUnitPlatform()
        testLogging {
            events("failed")
            exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
        }
    }
}
