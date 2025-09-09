## Run Command

```java
package com.levelrin;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Objects;

final public class Main {

    public static void main(final String... args) throws IOException, InterruptedException, URISyntaxException {
        final Path dirPath = Paths.get(
            Objects.requireNonNull(
                Main.class.getClassLoader().getResource("")
            ).toURI()
        );
        // It will print the output to System.out/System.err.
        final int exitCode = new ProcessBuilder("git", "--version")
            .directory(dirPath.toFile())
            .inheritIO().start().waitFor();
        if (exitCode != 0) {
            throw new RuntimeException("git --version failed with exit code " + exitCode);
        }
    }

}
```
