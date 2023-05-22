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
```

FYI, here are the imports:
```java
import com.amazonaws.services.dynamodbv2.AmazonDynamoDB;
import com.amazonaws.services.dynamodbv2.AmazonDynamoDBClientBuilder;
import com.amazonaws.services.dynamodbv2.document.DynamoDB;
import com.amazonaws.services.dynamodbv2.document.Item;
import com.amazonaws.services.dynamodbv2.document.Table;
```
