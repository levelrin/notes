## Read a file from the resources folder

```java
package com.levelrin;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public final class Main {

    public static void main(final String... args) throws IOException, URISyntaxException {
        // The file is located in the resources folder: src/main/resources/yoi.txt
        final Path path = Paths.get(ClassLoader.getSystemResource("yoi.txt").toURI());
        final String text = Files.readString(path, StandardCharsets.UTF_8);
        System.out.println(text);
    }

}
```
