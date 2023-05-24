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
