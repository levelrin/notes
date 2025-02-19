## Command Line Application

`build.gradle`:
```gradle
jar {
    manifest {
        attributes 'Main-Class': 'com.levelrin.Main'
    }
    // Fat jar configuration.
    from {
        configurations.runtimeClasspath.collect { it.isDirectory() ? it : zipTree(it) }
    }
    duplicatesStrategy = DuplicatesStrategy.EXCLUDE
    archiveFileName = "${project.name}-${project.version}-java${JavaVersion.current().toString()}.jar"
}
```

Compile:
```sh
./gradlew jar
```

Run:
```sh
java -jar build/libs/{jar-file-name}
```
