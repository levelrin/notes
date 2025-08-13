## When DOM is ready

Do something when the DOM is ready, but not necessarily for CSS, images, etc.

```js
document.addEventListener("DOMContentLoaded", () => {
    console.log("DOM is ready.");
});
```

## When page is fully loaded

Do something when the page is fully loaded, including CSS, images, etc.

```js
window.addEventListener("load", () => {
    console.log("Everything is loaded.");
});
```
