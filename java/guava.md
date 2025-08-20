## Read a resource

We want to read a file in the resources folder.

For example, we will read this file: `src/main/resources/static/index.html`.

Gradle Dependency:
```groovy
dependencies {
    implementation("com.google.guava:guava:33.4.8-jre")
}
```

Java:
```java
package com.levelrin;

import com.google.common.io.Resources;
import java.io.IOException;
import java.nio.charset.StandardCharsets;

final public class Main {

    public static void main(final String... args) throws IOException {
        final String content = Resources.toString(Resources.getResource("static/index.html"), StandardCharsets.UTF_8);
        System.out.println(content);
    }

}
```
