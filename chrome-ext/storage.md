## Counter Sample

This is the `manifest.json`:
```json
{
  "manifest_version": 3,
  "version": "0.1.0",
  "name": "Friendly Automator",
  "description": "Automate your tasks.",
  "icons": {
    "16": "images/icons/icon-16.png",
    "32": "images/icons/icon-32.png",
    "48": "images/icons/icon-48.png",
    "128": "images/icons/icon-128.png"
  },
  "action": {
    "default_popup": "popup/popup.html"
  },
  "permissions": ["storage"]
}
```

This is the relevant part:
```
"permissions": ["storage"]
```

This is the `popup/popup.html`:
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Title</title>
    <link rel="stylesheet" href="popup.css">
</head>
<body>
<h1>Count: <span id="count">0</span></h1>
<button type="button" id="incrementButton">Increment</button>
<script src="popup.js"></script>
</body>
</html>
```

This is the `popup/popup.js`:
```js
/**
 * Obtain the count and do something with it.
 * @param onCount A function that handles the result.
 *                The result (number) would be used as a parameter for this function.
 */
function runWithCount(onCount) {
    // We need to specify keys to retrieve data.
    // Ex: ["key1", "key2"].
    // The storage API will create a JSON with those keys and values as a result.
    chrome.storage.sync.get(["count"], function(result) {
        // Note that we need to specify the key again.
        onCount(result.count);
    });
}

/**
 * After the save, it will display the count.
 * @param count We will save this.
 */
function saveCount(count) {
    // The data we save must be in JSON.
    chrome.storage.sync.set({count: count}, function() {
        displayCount(count);
    });
}

function displayCount(count) {
    const spanCount = document.getElementById("count");
    spanCount.innerText = count;
}

document.addEventListener('DOMContentLoaded', () => {
    const incrementButton = document.getElementById('incrementButton');
    incrementButton.onclick = function() {
        runWithCount((currentCount) => {
            saveCount(currentCount + 1);
        });
    }
    // Display the count at the beginning.
    chrome.storage.sync.get(["count"], function(result) {
        if ("count" in result) {
            displayCount(result.count);
        } else {
            saveCount(0);
        }
    });
});
```

In this example, we use `storage.sync`.

However, we can also use `storage.local` or `storage.session` without changing method calls because they all have the same interface (methods).

The `storage.sync` is good for storing user preferences because:
 - The data is shared across all devices as long as the user is signed in to the same account.
 - The data is preserved even after the browser reboot.
 - It has the lowest quota (~100KB total).

The `storage.local` is good for general persistent data because:
 - The data is preserved even after the browser reboot.
 - It has a larger quota (around 10 MB, but it can be increased by requesting the "unlimitedStorage" permission).
 - The data is not shared across devices (only for that browser).

The `storage.session` is good for temporary workflows because:
 - The data is cleared when the browser is closed.
 - The data is not shared across devices.
 - It has a larger quota (around 10 MB).
