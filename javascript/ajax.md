## GET

```js
const ajax = new XMLHttpRequest();
ajax.open(
    "GET",
    "http://localhost:4567/yoi",
    // async or not
    true
);
// It helps browser parsing the response body.
// By default, it's plain text.
// By specifying the response type, you don't have to parse it to JSON manually.
// It does not affect the HTTP request.
ajax.responseType = "json";
ajax.onload = function() {
    if (ajax.status === 200) {
        const responseBody = ajax.response;
        // Assuming the value is {"one":"uno"}
        alert(responseBody.one);
    } else {
        console.error("Request failed. Status code: " + ajax.status);
    }
}
ajax.onerror = function() {
    console.error("Request failed due to a network error.");
}
ajax.send();
```

## POST

```js
const ajax = new XMLHttpRequest();
ajax.open(
    "POST",
    "http://localhost:4567/yoi",
    true
);
// We can have multiple headers by calling setRequestHeader() multiple times.
ajax.setRequestHeader("Content-Type", "application/json");
ajax.setRequestHeader("Authorization", "Bearer abc...");
ajax.onload = function() {
    if (ajax.status === 204) {
        alert("Request succeeded.");
    } else {
        console.error("Request failed. Status code: " + ajax.status);
    }
}
// Represent the JSON {"one":"uno"}
const rawPayload = {
    one: "uno"
}
// We need to convert the raw object into JSON string.
// The server may receive "[object Object]" if we send the raw one.
const jsonPayload = JSON.stringify(rawPayload);
ajax.send(jsonPayload);
```

## CORS

When we send an ajax request to a different origin, the browser will send a [preflight request](https://developer.mozilla.org/en-US/docs/Glossary/Preflight_request), in which the HTTP method would be `OPTIONS`.

For that reason, the server must be ready to accept the OPTIONS request.

The response to the OPTIONS request must have the following response headers:
1. [Access-Control-Allow-Origin](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Origin)
    - The value would be the origin of ajax requests.
    - It should include scheme but exclude path.
    - Good ex: http://localhost:4567
    - Bad ex: http://localhost:4567/yoi or localhost:4567
    - We cannot specify multiple values.
    - However, we can use the wildcard \(\*\) to allow any origin.
    - In that case, the header would look like this: `Access-Control-Allow-Origin: *`.
3. [Access-Control-Allow-Headers](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Headers) if ajax sets headers.
    - For example, if the ajax sets `Content-Type` and `Authorization` headers, the response must have a header like this `Access-Control-Allow-Headers: Content-Type, Authorization`.
    - The server don't have to worry about the request headers automatically set by the browser.
4. [Access-Control-Allow-Methods](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Methods)
    - Ex: `Access-Control-Allow-Methods: GET, POST` or `Access-Control-Allow-Methods: *`
    - Although it may not be required depending on the browser, it's better to specify just in case.

FYI, there are more CORS-related response headers to restrict requests.

Note that the server must include the `Access-Control-Allow-Origin` header for the main endpoint as well.

For example, if the ajax sends a POST request to the server, the browser will send an OPTIONS request to the endpoint first.

Subsequenlty, the browser will send the POST request by the ajax.

As a result, the server must respond to both OPTIONS and POST requests with the same `Access-Control-Allow-Origin` header.
