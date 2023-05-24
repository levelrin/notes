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
