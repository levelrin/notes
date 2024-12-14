When the button on the `popup/popup.html` is clicked, the following will happen:
1. Visit https://www.google.com on a new tab.
2. Type "test" on the search bar.
3. Press enter to see the search result.

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
  "permissions": ["tabs", "scripting"],
  "host_permissions": [
    "http://*/*",
    "https://*/*"
  ],
  "background": {
    "service_worker": "background.js"
  }
}

```

This is the relevant part:
```
"permissions": ["tabs", "scripting"],
"host_permissions": [
  "http://*/*",
  "https://*/*"
],
"background": {
  "service_worker": "background.js"
}
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
<button type="button" id="search-button">Search Google</button>
<script src="popup.js"></script>
</body>
</html>
```

This is the `popup/popup.js`:
```js
window.onload = () => {
    const searchButton = document.getElementById("search-button");
    searchButton.onclick = () => {
        // Send a message to background processor.
        // The `background.js` will handle the message.
        chrome.runtime.sendMessage({about: "search"});
    }
}
```

This is the `background.js`:
```js
const tabIds = new Set();

chrome.runtime.onMessage.addListener((message, __, sendResponse) => {
    if (message.about === "search") {
        chrome.tabs.create({
            url: "https://www.google.com"
        }).then((tab) => {
            tabIds.add(tab.id);
        });
    }
});

chrome.tabs.onUpdated.addListener((tabId, changeInfo, ___) => {
    if (tabIds.has(tabId) && changeInfo.status === "complete") {
        chrome.scripting.executeScript({
            target: { tabId: tabId },
            files: ["scripts/search.js"]
        }).then(() => {
            tabIds.delete(tabId);
        });
    }
});
```

This is the `scripts/search.js`:
```js
const textarea = document.getElementsByTagName("textarea")[0];
textarea.value = "test";

// Create a KeyboardEvent for pressing Enter.
textarea.dispatchEvent(
    new KeyboardEvent('keydown', {
        key: 'Enter',
        code: 'Enter',
        keyCode: 13,
        which: 13,
        bubbles: true
    })
);
```
