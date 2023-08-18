## Prepare for the authentication to use `aws` command

1. Go to `IAM` on AWS website.
2. Go to `Users` tab under `Access management`.
3. Add a user.
4. Edit the user like this:
    1. Add the permission for using lambda. Ex: `AWSLambda_FullAccess`.
    2. Create an access key from the `Security credentials` tab. Take a note of the access key ID and secret access key.
5. Go to terminal and configure the `aws` cli like this:
    ```sh
    aws configure
    ```

## Deploy the code

Gradle configuration (build.gradle):
```groovy
tasks.register('buildZip', Zip) {
    from compileJava
    from processResources
    into('lib') {
        from configurations.runtimeClasspath
    }
}
```

Run the Gradle task `buildZip`:
```sh
./gradlew buildZip
```

Upload the zip file:
```sh
aws lambda update-function-code --function-name <lambda function name> --zip-file fileb://<path to zip file>
```

Ex:
```sh
aws lambda update-function-code --function-name note-front --zip-file fileb://./build/distributions/lambda-playground-0.0.1.zip
```

## Specify the entrypoint of lambda function

We need to tell the lambda which method it should run.

1. Go to the lambda configuration page on AWS website.
2. Go to the `Code` tab and edit the `Runtime settings`.
3. Set the value of `Handler`. Ex: com.levelrin.YoiHandler::handleRequest

## HTTP Communication

Add the following Gradle dependencies:
```groovy
implementation 'com.amazonaws:aws-lambda-java-core:1.2.2'
implementation 'com.amazonaws:aws-lambda-java-events:3.11.1'
```

Create a class for the lambda like this:
```java
import com.amazonaws.services.lambda.runtime.Context;
import com.amazonaws.services.lambda.runtime.RequestHandler;
import com.amazonaws.services.lambda.runtime.events.APIGatewayV2HTTPEvent;
import com.amazonaws.services.lambda.runtime.events.APIGatewayV2HTTPResponse;
import java.util.HashMap;
import java.util.Map;

public final class YoiHandler implements RequestHandler<APIGatewayV2HTTPEvent, APIGatewayV2HTTPResponse> {

    @Override
    public APIGatewayV2HTTPResponse handleRequest(final APIGatewayV2HTTPEvent event, final Context context) {
        final APIGatewayV2HTTPResponse response = new APIGatewayV2HTTPResponse();
        response.setIsBase64Encoded(false);
        response.setStatusCode(200);
        final Map<String, String> headers = new HashMap<>();
        headers.put("Content-Type", "text/html");
        response.setHeaders(headers);
        response.setBody(
            """
            <!DOCTYPE html>
            <html>
            <head>
            <title>Yoi Yoi</title>
            </head>
            <body>
                <h1>Yoi Yoi</h1>
                <p>Lambda function is working!</p>
            </body>
            </html>
            """
        );
        return response;
    }

}
```

## Logging

```java
@Override
public APIGatewayV2HTTPResponse handleRequest(final APIGatewayV2HTTPEvent event, final Context context) {
    final LambdaLogger logger = context.getLogger();
    // You will see this log in AWS CloudWatch.
    logger.log("Yoi Yoi from lambda logger!");
}
```

## Access DynamoDB

The lambda's execution role must have the permissions to access DynamoDB.

1. Go to `IAM` on AWS website.
2. Go to `Roles` tab under `Access management`.
3. Click the role used by the lambda.
4. Add permissions to grant access. Ex: AmazonDynamoDBFullAccess

Gradle dependency:
```groovy
implementation 'com.amazonaws:aws-java-sdk-dynamodb:1.12.470'
```

Sample Java code:
```java
// Create a database object.
// We don't have to worry about the authentication as long as it's executed by the lambda.
// By the way, we can access it outside of lambda as long as the credentials are configured in the environment.
// More info at https://docs.aws.amazon.com/sdk-for-java/v1/developer-guide/credentials.html
// Note that we must add permissions to the user corresponds to the credentials via IAM.
final AmazonDynamoDB dynamoClient = AmazonDynamoDBClientBuilder.standard().build();
final DynamoDB db = new DynamoDB(dynamoClient);

// Get an item from the database.
final Table users = db.getTable("users");
final Item item = users.getItem("username", "test01");
final String note = item.getString("note");

// Check if the item exists in the table.
final Item item2 = users.getItem("username", "test02");
if (item2 == null) {
    // Item does not exist.
} else {
    // Item exists.
}

// Put a new item into the table.
final Item item3 = new Item()
    .withPrimaryKey("username", "test03")
    .withString("note", "tres");
// Note that it will overwrite the value if the item exists.
users.putItem(item3);

// Update an item.
final Map<String, String> attributePlaceholders = new HashMap<>();
attributePlaceholders.put("#N", "note");
final Map<String, Object> valuePlaceholders = new HashMap<>();
valuePlaceholders.put(":note", "one");
// Note that it will create a new item if it does not exist.
// The difference between putItem and updateItem method is that
// putItem only changes the specified attributes, whereas
// putItem replaces the entire item. Any unspecified attributes will be removed.
users.updateItem(
    // key attribute name
    "username",
    // key attribute value
    "test01",
    // update expression
    "SET #N = :note",
    attributePlaceholders,
    valuePlaceholders
);
```

FYI, here are the imports:
```java
import com.amazonaws.services.dynamodbv2.AmazonDynamoDB;
import com.amazonaws.services.dynamodbv2.AmazonDynamoDBClientBuilder;
import com.amazonaws.services.dynamodbv2.document.DynamoDB;
import com.amazonaws.services.dynamodbv2.document.Item;
import com.amazonaws.services.dynamodbv2.document.Table;
```

## CORS

Make sure the following:
1. CORS is enabled in API Gateway
    - The responder's endpoint should be configured, not the sender's.
    - We should configure the response header values in the `Integration Response`.
    - We can check if we configured correctly by sending an HTTP OPTIONS request like this:
     ```sh
     curl -v -X OPTIONS 'https://a1b2c3.execute-api.us-east-1.amazonaws.com/default/note-save'
     ```
2.  The lambda also needs to include CORS headers in its response.

## Invoke another lambda function asynchronously

The invoker needs the `AWSLambdaRole` policy in `IAM`.

The invoker needs the following dependency:
```groovy
implementation 'software.amazon.awssdk:apigatewaymanagementapi:2.20.74'
```

Invoke a lambda function like this:
```java
// Invoke lambda function asynchronously without waiting.
final AWSLambda lambda = AWSLambdaAsyncClientBuilder.standard().build();
lambda.invoke(
    new InvokeRequest()
        .withFunctionName("async-invokee")
        .withInvocationType(InvocationType.Event)
        .withPayload(
            """
            {"name":"test01", "fruit":"apple"}
            """
        )
);
```

Here is the code for invokee.
Handle the request like this:
```java
// We use RequestStreamHandler because the lambda may automatically deserialize the input and fail.
// In other words, we want to make input deserialization our responsibility to avoid possible failure by lambda.
public final class AsyncInvokee implements RequestStreamHandler {

    @Override
    public void handleRequest(final InputStream input, final OutputStream output, final Context context) throws IOException {
        try (input; output) {
            final LambdaLogger logger = context.getLogger();
            final JsonObject json = new Gson().fromJson(
                new String(input.readAllBytes(), StandardCharsets.UTF_8),
                JsonObject.class
            );
            final String name = json.get("name").getAsString();
            final String fruit = json.get("fruit").getAsString();
            logger.log(
                String.format(
                    "name: %s, fruit: %s",
                    name, fruit
                )
            );
        }
    }

}
```

## WebSocket

We suggest creating lambdas before setting the routes in API Gateway to prevent a possible scenario in that API Gateway could not find the lambdas even if we configure it correctly for some reason.

We need to make sure that the lambda's role has permissions, such as `AmazonAPIGatewayInvokeFullAccess`, that allows sending messages to clients.

In API Gateway, we can see `WebSocket URL` and `Connection URL` in the stage.

Ex:
```
WebSocket URL: wss://a1b2c3.execute-api.us-east-1.amazonaws.com/production
Connection URL: https://a1b2c3.execute-api.us-east-1.amazonaws.com/production/@connections
```

`WebSocket URL` is for establishing a connection. It is typically used by the client when they connect to the server.

`Connection URL` is for sending a message to the connection. It is typically used by the server when it sends messages to the client. Note that we will exclude `@connection` from the URL in this example.

Let's say `Route Selection Expression` is set to `$request.body.action` in JSON.

The client needs to include the route information in its payload like this:
```json
{
    "action":"sendMessage",
    "message":"Yoi Yoi"
}
```

The `message` is a custom attribute, and we expect the server to parse the payload to get the message.

FYI, API Gateway will transform the payload with more information like this:
```json
{
    "requestContext": {
        "routeKey": "sendMessage",
        "messageId": "abcde=",
        "eventType": "MESSAGE",
        "extendedRequestId": "abcdefg=",
        "requestTime": "29/May/2023:00:40:39 +0000",
        "messageDirection": "IN",
        "stage": "demo",
        "connectedAt": 1234567890123,
        "requestTimeEpoch": 3210987654321,
        "identity": {
            "userAgent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36",
            "sourceIp": "123.456.78.901"
        },
        "requestId": "abcdefg=",
        "domainName": "a1b2c3.execute-api.us-east-1.amazonaws.com",
        "connectionId": "abcdefghijk=",
        "apiId": "a1b2c3"
    },
    "body": "{\"action\":\"sendMessage\",\"message\":\"Yoi Yoi\"}",
    "isBase64Encoded": false
}
```

This information might be useful for debugging.

However, API Gateway's log is off by default. We need to enable log to see the transformed payload.

By the way, the server doesn't have to parse the entire JSON. It just need to parse the `body` to get the message.

Here is the sample code for lambda:
```java
package com.levelrin;

import com.amazonaws.services.lambda.runtime.Context;
import com.amazonaws.services.lambda.runtime.RequestHandler;
import com.amazonaws.services.lambda.runtime.events.APIGatewayV2WebSocketEvent;
import com.amazonaws.services.lambda.runtime.events.APIGatewayV2WebSocketResponse;
import com.google.gson.Gson;
import com.google.gson.JsonObject;
import java.net.URI;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import software.amazon.awssdk.core.SdkBytes;
import software.amazon.awssdk.services.apigatewaymanagementapi.ApiGatewayManagementApiClient;
import software.amazon.awssdk.services.apigatewaymanagementapi.model.PostToConnectionRequest;

/**
 * Sample chat application.
 */
public final class Chat implements RequestHandler<APIGatewayV2WebSocketEvent, APIGatewayV2WebSocketResponse> {

    /**
     * We will store connection IDs and their corresponding information to facilitate communication.
     * Actually, we should use an external storage such as DynamoDB, RDS, or ElastiCache.
     * We should never rely on this kind of field for production
     * because the data will be cleared when the lambda is not used or some updated code is deployed.
     * That being said, the data will persist temporarily, and we use this map for demonstration purposes only.
     * Key: connection id
     * Value: username
     */
    private final Map<String, String> users = new ConcurrentHashMap<>();

    @Override
    public APIGatewayV2WebSocketResponse handleRequest(final APIGatewayV2WebSocketEvent event, final Context context) {
        // It's related to "Route Selection Expression" defined by API Gateway.
        // For example, let's say "Route Selection Expression" is set to "$request.body.action" in JSON.
        // That means the client should send a data like this: {"action": "sendMessage", "message": "Yoi Yoi"}
        // The value of "action", which is "sendMessage" in this case, will be the route key.
        final String routeKey = event.getRequestContext().getRouteKey();
        final String connectionId = event.getRequestContext().getConnectionId();
        // We don't have to have conditions based on the route key like this if each route uses different lambdas.
        // In this case, we just use the same lambda (this class) for all routes.
        // $connect, $disconnect, and $default are given by API Gateway.
        // The client doesn't have to worry about them.
        // The client needs to include custom route key ("sendMessage" in this case) in its payload, though.
        if ("$connect".equals(routeKey)) {
            // URL query string is only available during connect.
            // We will get null on event.getQueryStringParameters() for other routes.
            final String username = event.getQueryStringParameters().get("username");
            // We will have the "error code GoneException" with the status code 410
            // if we broadcast the message after putting self into the map.
            // The reason is that the self's connection is not established yet until the lambda returns a response.
            this.broadcast(
                String.format("%s has joined the chat.", username)
            );
            this.users.put(connectionId, username);
        } else if ("$disconnect".equals(routeKey)) {
            this.users.remove(connectionId);
            final String username = this.users.get(connectionId);
            this.broadcast(
                String.format("%s has left the chat.", username)
            );
        } else if ("$default".equals(routeKey)) {
            // Any unknown route will end up here.
        } else if ("sendMessage".equals(routeKey)) {
            final String username = this.users.get(connectionId);
            // event.getBody() will give us the "body" that looks like this: {"action": "sendMessage", "message": "Yoi Yoi"}
            // Note that the route key ("action" in this case) is in the body.
            // For that reason, we need to parse the JSON to get the information we want ("message" in this case).
            final String message = new Gson().fromJson(event.getBody(), JsonObject.class).get("message").getAsString();
            this.broadcast(
                String.format("%s: %s", username, message)
            );
        } else {
            // Any unknown route will go to the $default route.
            // In other words, this block is not reachable in theory.
        }
        final APIGatewayV2WebSocketResponse response = new APIGatewayV2WebSocketResponse();
        // We must return a successful response.
        response.setStatusCode(204);
        return response;
    }

    /**
     * Send a message to all clients.
     * @param message Clients will receive it.
     */
    private void broadcast(final String message) {
        try (
            ApiGatewayManagementApiClient client = ApiGatewayManagementApiClient
            .builder()
            .endpointOverride(
                URI.create(
                    // Connection URL, not the WebSocket URL
                    "https://a1b2c3.execute-api.us-east-1.amazonaws.com/demo"
                )
            )
            .build()
        ) {
            for (final String connectionId : this.users.keySet()) {
                final PostToConnectionRequest request = PostToConnectionRequest
                    .builder()
                    .connectionId(
                        connectionId
                    )
                    .data(SdkBytes.fromUtf8String(message))
                    .build();
                // We should not call this for the connection id that is still on the $connect route
                // because the connection is not established yet.
                client.postToConnection(request);
            }
        }
    }

}
```

Gradle dependencies:
```groovy
implementation 'com.amazonaws:aws-lambda-java-core:1.2.2'
implementation 'com.amazonaws:aws-lambda-java-events:3.11.1'
implementation 'software.amazon.awssdk:apigatewaymanagementapi:2.20.74'

// For parsing JSON. You don't have to use it if you want to use other tools.
implementation 'com.google.code.gson:gson:2.10.1'
```

Here is the sample code for clients:
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Chat Demo</title>
</head>
<body>
<label for="name">Name</label>
<input type="text" id="name" name="name" required minlength="4" maxlength="8" size="10">
<br>
<label for="message">Message</label>
<input type="text" id="message" name="message" required minlength="4" maxlength="100" size="20">
<button onclick="sendMessage()">Send</button>
<script>
    // I know it's a bad practice :P
    let ws = null;
    function sendMessage() {
        const username = document.getElementById("name").value;
        if (username === "") {
            alert("Please put your username.");
            return;
        }
        const input = document.getElementById("message").value;
        if (input === "") {
            alert("Please put your message.");
            return;
        }
        if (ws == null) {
            // WebSocket URL, not the Connection URL.
            // We use URL query string to pass the username.
            // Although there might be some hack, javascript does not support adding custom headers at the time of writing.
            ws = new WebSocket("wss://a1b2c3.execute-api.us-east-1.amazonaws.com/demo?username=" + username);
            ws.addEventListener("open", (event) => {
                console.log("Connected!");
                ws.send(
                    // Client needs to sepcify the route key.
                    // The server will need to parse the payload to get the target information.
                    JSON.stringify({
                        action: "sendMessage",
                        message: input
                    })
                );
            });
            ws.addEventListener("message", (event) => {
                console.log(event.data);
            });
        } else {
            ws.send(
                JSON.stringify({
                    action: "sendMessage",
                    message: input
                })
            );
        }
    }
</script>
</body>
</html>
```
