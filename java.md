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
}
```

Compile:
```sh
./gradlew jar
```

Run:
```sh
java -jar build/libs/project-name-0.1.0.jar
```
