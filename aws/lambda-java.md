## Deploy the code

Gradle configuration (build.gradle):
```groovy
tasks.register('buildZip', Zip) {
    from compileJava
    from processResources
    into('lib') {
        from configurations.runtimeClasspath
    }
}
```

Run the Gradle task `buildZip`:
```sh
./gradlew buildZip
```

Upload the zip file:
```sh
aws lambda update-function-code --function-name <lambda function name> --zip-file fileb://<path to zip file>
```

Ex:
```sh
aws lambda update-function-code --function-name note-front --zip-file fileb://./build/distributions/lambda-playground-0.0.1.zip
```
