## Gradle

```groovy
dependencies {
    implementation 'com.jayway.jsonpath:json-path:2.9.0'
}
```

## Simple Parse

```java
package com.levelrin;

import com.jayway.jsonpath.JsonPath;
import java.util.List;

public final class Main {

    public static void main(final String... args) {
        final String json = """
            {
              "one": "uno",
              "fruits": ["apple", "banana", "orange"]
            }
            """;
        final String one = JsonPath.read(json, "$.one");
        final List<String> fruits = JsonPath.read(json, "$.fruits");
        final String apple = JsonPath.read(json, "$.fruits[0]");
    }

}
```
