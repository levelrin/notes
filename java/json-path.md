## Gradle

```groovy
dependencies {
    implementation 'com.jayway.jsonpath:json-path:2.9.0'
    // Optional logging.
    implementation 'org.slf4j:slf4j-simple:2.0.13'
}
```

## Logging

`src/main/resources/simplelogger.properties`:
```properties
# Set the application's log level.
org.slf4j.simpleLogger.defaultLogLevel=DEBUG
org.slf4j.simpleLogger.dateTimeFormat=yyyy-MM-dd HH:mm:ss.SSS zzz
org.slf4j.simpleLogger.showThreadName=true
org.slf4j.simpleLogger.showLogName=true
org.slf4j.simpleLogger.showShortLogName=false
org.slf4j.simpleLogger.showDateTime=true

# Set the log level of JsonPath.
org.slf4j.simpleLogger.log.com.jayway.jsonpath=WARN
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

## Merge JSON

Let's say we have two JSONs: `{"one":"uno"}` and `{"two":"dos"}`.

We want to merge them so that the result would look like this: `{"one":"uno","two":"dos"}`.

Here's how to do it:
```java
package com.levelrin;

import com.jayway.jsonpath.DocumentContext;
import com.jayway.jsonpath.JsonPath;
import java.util.Map;

public final class Main {

    public static void main(final String... args) {
        final DocumentContext jsonPart = JsonPath.parse("{}");
        jsonPart.put("$", "one", "uno");
        final DocumentContext jsonWhole = JsonPath.parse("{}");
        jsonWhole.put("$", "two", "dos");
        final Map<String, Object> jsonPartMap = jsonPart.json();
        for (Map.Entry<String, Object> entry : jsonPartMap.entrySet()) {
            jsonWhole.put("$", entry.getKey(), entry.getValue());
        }
        System.out.println(jsonWhole.jsonString());
    }

}
```

## Check if path exists

```java
package com.levelrin;

import com.jayway.jsonpath.DocumentContext;
import com.jayway.jsonpath.JsonPath;
import com.jayway.jsonpath.PathNotFoundException;

public final class Main {

    public static void main(final String... args) {
        final String raw = """
            {
              "one": "uno",
              "fruits": ["apple", "banana", "orange"]
            }
            """;
        final DocumentContext json = JsonPath.parse(raw);
        // Unfortunately, JsonPath does not provide a method to check if the key exists at the time of writing.
        // Although it's ugly, we may need to exploit the exception.
        try {
            final String two = json.read("$.two");
            System.out.println(two);
        } catch (final PathNotFoundException ex) {
            System.out.println("two does not exist.");
        }
    }

}
```
