plugins {
    application
}

dependencies {
    implementation(project(":core"))
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.9.0")
}

application {
    mainClass.set("dev.simplesemantic.cli.MainKt")
    applicationName = "simple-semantic"
}

tasks.named<JavaExec>("run") {
    standardInput = System.`in`
}
