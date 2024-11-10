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

## Modify JSON

```java
package com.levelrin;

import com.jayway.jsonpath.DocumentContext;
import com.jayway.jsonpath.JsonPath;

public final class Main {

    public static void main(final String... args) {
        final String json = """
            {
              "one": "uno",
              "fruits": ["apple", "banana", "orange"]
            }
            """;
        final DocumentContext jsonContext = JsonPath.parse(json);
        // Add a new element from the root.
        jsonContext.put("$", "username", "test01");
        // Replace an element.
        jsonContext.put("$", "one", "ichi");
        // Delete elements if the value equals to "apple".
        // ?(<expression>) is the conditioned filter.
        // @ means the current node (position).
        jsonContext.delete("$.fruits[?(@ == 'apple')]");
        System.out.println(jsonContext.jsonString());
    }

}
```
