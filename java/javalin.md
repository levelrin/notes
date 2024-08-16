## build.gradle

```groovy
plugins {
    id 'application'
    id 'java'
}

mainClassName = 'com.levelrin.Main'

dependencies {
    implementation 'io.javalin:javalin:6.2.0'
    implementation 'org.slf4j:slf4j-simple:2.0.13'
    implementation 'com.fasterxml.jackson.core:jackson-databind:2.17.0'
}
```

## Logging

`src/main/resources/simplelogger.properties`:
```properties
# Set the application's log level.
org.slf4j.simpleLogger.defaultLogLevel=INFO
org.slf4j.simpleLogger.dateTimeFormat=yyyy-MM-dd HH:mm:ss.SSS zzz
org.slf4j.simpleLogger.showThreadName=true
org.slf4j.simpleLogger.showLogName=true
org.slf4j.simpleLogger.showShortLogName=false
org.slf4j.simpleLogger.showDateTime=true

# Set the log level of Javalin.
org.slf4j.simpleLogger.log.io.javalin=WARN
org.slf4j.simpleLogger.log.org.eclipse.jetty=WARN
```

Write log:
```java
package com.levelrin;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Main {

    private static final Logger LOGGER = LoggerFactory.getLogger(Main.class);

    public static void main(String[] args) {
        final String var1 = "Hello";
        final String var2 = "World";
        if (LOGGER.isInfoEnabled()) {
            LOGGER.info("Some message. var1: {}, var2: {}", var1, var2);
        }
    }

}
```

## Static Files

`resources/static/index.html`:
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Title</title>
</head>
<body>
Yoi Yoi
</body>
</html>
```

Main class:
```java
package com.levelrin;

import io.javalin.Javalin;

public class Main {

    public static void main(final String... args) {
        Javalin.create(config -> config.staticFiles.add("/static")).start(7070);
    }

}
```
