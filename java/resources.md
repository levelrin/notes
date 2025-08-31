## Read a file from the resources

```java
package com.levelrin;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Objects;

final public class Main {

    public static void main(final String... args) {
        final String content;
        final String resourcePath = "static/index.html";
        final ClassLoader classLoader = Main.class.getClassLoader();
        try {
            content = Files.readString(
                Paths.get(
                    Objects.requireNonNull(
                        classLoader.getResource(resourcePath)
                    ).toURI()
                ),
                StandardCharsets.UTF_8
            );
        } catch (final IOException | URISyntaxException ex) {
            throw new IllegalStateException("Failed to read the resource. Resource path: " + resourcePath, ex);
        }
        System.out.println(content);
    }

}
```

## Check if resource exists

```java
package com.levelrin;

import java.net.URL;

final public class Main {

    public static void main(final String... args) {
        final String resourcePath = "static/index.html";
        final ClassLoader classLoader = Main.class.getClassLoader();
        final URL resourceUrl = classLoader.getResource(resourcePath);
        if (resourceUrl == null) {
            System.out.println("Resource not found :(");
        } else {
            System.out.println("Resource found!");
        }
    }

}
```
