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

## Handle HTML Form

`resources/static/index.html`:
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Login</title>
</head>
<body>

<form action="/login" method="POST">
    <label for="username">Username</label>
    <!-- The `name` attribute will be used for `context.formParam(key)` in Javalin. -->
    <input type="text" id="username" name="username" aria-describedby="username" required>

    <label for="password">Password</label>
    <input type="password" id="password" name="password" aria-describedby="password" required>

    <button type="submit">Submit</button>
</form>

</body>
</html>
```

`resources/static/home.html`:
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Home</title>
</head>
<body>
Hello, <span id="username-label"></span>!
<script src="script.js"></script>
</body>
</html>
```

`resources/static/script.js`:
```js
window.onload = function() {
    if (hasCookie("username")) {
        const username = cookie("username");
        document.getElementById("username-label").innerText = username;
        // The main client code should go here.
    } else {
        // It's for when the user tries to directly access this URL without going through the login page.
        alert("Please login first.");
        window.location.pathname = "/index.html";
    }
}

/**
 * Get the cookie value.
 * @param name Key.
 * @returns {string}
 */
function cookie(name) {
    let cookieArr = document.cookie.split(";");
    for(let i = 0; i < cookieArr.length; i++) {
        let cookiePair = cookieArr[i].split("=");
        if(name === cookiePair[0].trim()) {
            return decodeURIComponent(cookiePair[1]);
        }
    }
    alert("Could not find the cookie name: " + name);
}

/**
 * Check if the cookie exists.
 * @param name Key.
 * @return {boolean}
 */
function hasCookie(name) {
    let cookieArr = document.cookie.split(";");
    for(let i = 0; i < cookieArr.length; i++) {
        let cookiePair = cookieArr[i].split("=");
        if(name === cookiePair[0].trim()) {
            return true;
        }
    }
    return false;
}
```

Main class:
```java
package com.levelrin;

import io.javalin.Javalin;
import io.javalin.http.HttpStatus;

public class Main {

    public static void main(final String... args) {
        Javalin.create(config -> config.staticFiles.add("/static"))
            .post("/login", context -> {
                final String username = context.formParam("username");
                final String password = context.formParam("password");

                // Guard Clause.
                if (username == null || password == null) {
                    // Failed to get the username and password.
                    // Set the status code.
                    context.status(HttpStatus.UNAUTHORIZED);
                    // Return this String as a response body.
                    context.result("Login Failed.");
                    return;
                }
                // Assume the credentials are validated.

                // Pass the username to the next page via cookie.
                context.cookie("username", username);
                context.redirect("/home.html", HttpStatus.MOVED_PERMANENTLY);
            })
            .start(7070);
    }

}
```

## Deal with JSON

```java
package com.levelrin;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import io.javalin.Javalin;
import java.util.ArrayList;
import java.util.List;

public class Main {

    public static void main(final String... args) {
        final ObjectMapper jackson = new ObjectMapper();
        Javalin.create()
            .post("/register-item", context -> {
                // It checks if the header "Content-Type: application/json" is used.
                // It does not check the actual data, though.
                if (context.isJson()) {
                    //  Let's say the request body looks like this:
                    // {"username": "test01", "items": ["apple", "banana", "orange"]}
                    final String rawRequestBody = context.body();

                    // Parse JSON.
                    final JsonNode jsonRequestBody = jackson.readTree(rawRequestBody);
                    final String username = jsonRequestBody.get("username").asText();
                    final List<String> items = new ArrayList<>();
                    jsonRequestBody.get("items").forEach(item -> items.add(item.asText()));

                    // Build JSON object.
                    // We will build a response body like this:
                    // {"username": "test01", "items": ["apple", "banana", "orange"]}
                    final ObjectNode jsonResponse = jackson.createObjectNode()
                        .put("username", username);
                    final ArrayNode array = jackson.createArrayNode();
                    array.add("apple");
                    array.add("banana");
                    array.add("orange");
                    jsonResponse.set("items", array);

                    // By the way, we can stringify a JSON object like this:
                    //final String rawResponseBody = jackson.writeValueAsString(jsonResponse);

                    // Set the response body.
                    context.json(jsonResponse);
                }
            })
            .start(7070);
    }

}
```

## Handle HTTP requests globally

```java
package com.levelrin;

import io.javalin.Javalin;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Main {

    private static final Logger LOGGER = LoggerFactory.getLogger(Main.class);

    public static void main(final String... args) {

        Javalin.create()
            .before(context -> {
                if (LOGGER.isInfoEnabled()) {
                    LOGGER.info("Received a request. IP: {}", context.ip());
                }
            })
            .get("/hello", context -> {
                if (LOGGER.isInfoEnabled()) {
                    LOGGER.info("Say hello to IP: {}", context.ip());
                }
                context.result("Hello World");
            })
            .after(context -> {
                if (LOGGER.isInfoEnabled()) {
                    LOGGER.info("Request handled. IP: {}", context.ip());
                }
            })
            .start(7070);
    }

}
```
