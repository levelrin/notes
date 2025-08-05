## Wait until the user stops editing

We want to run a function when the textarea is edited.

However, we don't want to run the function immediately because the user might still be typing.

So, we want to wait for a few seconds until the user stops editing.

By the way, this scenario is referred to as `debouncing`.

```js
const textarea = document.getElementById("our-textarea");
let debounceTimeout;
textarea.addEventListener("input", () => {
    clearTimeout(debounceTimeout);
    debounceTimeout = setTimeout(() => {
        const value = textarea.value;
        console.log("User stopped typing. Value is:", value);
    }, 2000);
});
```

By the way, the same approach works for the plain textfield (the `<input type="text">` tag).
