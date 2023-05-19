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
