## Read all resources

Dependency:
```groovy
dependencies {
    implementation 'io.github.classgraph:classgraph:4.8.180'
}
```

Java:
```java
package com.levelrin;

import io.github.classgraph.ClassGraph;
import io.github.classgraph.Resource;
import io.github.classgraph.ScanResult;
import java.io.IOException;

final public class Main {

    public static void main(final String... args) {
        // ClassGraph can get all resources including the following:
        //  - Files in the `src/main/resources` directory.
        //  - Files in the `src/main/java` directory.
        //  - Files of the libraries.
        // It can find resources even if the application is executed as a fat-jar,
        // which is why we may want to use GlassGraph.
        // If we use `new ClassGraph().acceptPaths("").scan()`, it will get all resources.
        // However, in this case, we are only interested in files in the `src/main/resources` directory.
        // For that reason, we created a directory in the `src/main/resources` directory,
        // so that all our resources would have unique paths.
        // In other words, we would place our resources in this directory: `src/main/resources/com/levelrin/demo/resources`.
        // Note that we just need to use path relative to the `src/main/resources` directory.
        try (ScanResult scanResult = new ClassGraph().acceptPaths("com/levelrin/demo/resources").scan()) {
            for (final Resource resource : scanResult.getAllResources()) {
                try (resource) {
                    final String path = resource.getPath();
                    final String content = resource.getContentAsString();
                    System.out.println("path: " + path);
                    System.out.println("content:\n" + content + "\n");
                } catch (final IOException ex) {
                    throw new IllegalStateException("Failed to read the resource " + resource, ex);
                }
            }
        }
    }

}
```
