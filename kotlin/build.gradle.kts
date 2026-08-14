import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import org.jetbrains.kotlin.gradle.dsl.KotlinJvmProjectExtension

plugins {
    kotlin("jvm") version "2.2.20" apply false
    kotlin("plugin.serialization") version "2.2.20" apply false
}

allprojects {
    group = "dev.simplesemantic"
    version = "0.1.0"

    repositories {
        mavenCentral()
    }
}

// JDK 22 is the floor: the Foreign Function & Memory API is final there, so
// java.lang.foreign needs no --enable-preview. Checked here rather than left to
// a confusing "cannot find symbol java.lang.foreign.Arena" from the compiler.
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
            // Bytecode targets 22 even when built on a newer JDK.
            jvmTarget.set(JvmTarget.JVM_22)
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
