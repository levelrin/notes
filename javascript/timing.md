## Wait until element is loaded

```js
function waitForElement(querySelector) {
    return new Promise(resolve => {
        // If the element is already there, resolve immediately.
        if (document.querySelector(querySelector)) {
            return resolve(document.querySelector(querySelector));
        }
        const observer = new MutationObserver(mutations => {
            const element = document.querySelector(querySelector);
            if (element) {
                resolve(element);
                // Stop looking once found.
                observer.disconnect();
            }
        });
        // Watch the entire document for added nodes.
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    });
}

// Usage:
waitForElement("#button").then((element) => {
    element.click();
});
```
