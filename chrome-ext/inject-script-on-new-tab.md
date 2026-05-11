When the button on the `popup/popup.html` is clicked, the following will happen:
1. Visit https://www.google.com on a new tab.
2. Type "test" in the search bar.
3. Press Enter to see the search result.

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
const activeTabsUpdateListeners = {};
function removeListener(tabId) {
    if (activeTabsUpdateListeners[tabId]) {
        chrome.tabs.onUpdated.removeListener(activeTabsUpdateListeners[tabId]);
        delete activeTabsUpdateListeners[tabId];
    }
};
function injectJs(currentTabId, scriptLocation) {
    const listener = (tabId, changeInfo) => {
        if (tabId === currentTabId && changeInfo.status === "complete") {
            // Once this listener is executed, remove it to avoid duplicated registration.
            removeListener(tabId);
            chrome.scripting.executeScript({
                target: {tabId: currentTabId},
                files: [scriptLocation]
            });
        }
    };
    activeTabsUpdateListeners[currentTabId] = listener;
    chrome.tabs.onUpdated.addListener(listener);
}
// Although listeners are supposed to remove themselves, they may fail.
// For such a case, we ensure to remove them by this logic to prevent memory leaks.
chrome.tabs.onRemoved.addListener((tabId) => {
    removeListener(tabId);
});
chrome.runtime.onMessage.addListener((message, sender, ___) => {
    if (message.about === "search") {
        chrome.tabs.create({url: "https://www.google.com"}).then((tab) => {
            injectJs(tab.id, "scripts/search.js");
        });
    } else if (message.about === "prepareAfterSearch") {
        injectJs(sender.tab.id, "scripts/afterSearch.js");
    }
});
```

This is the `scripts/search.js`:
```js
const textarea = document.getElementsByTagName("textarea")[0];
textarea.value = "test";

// Before we hit enter, we should add a listener to inject JS after the page reload.
// It's important to note that the code in here is only effective in the current URL.
chrome.runtime.sendMessage({about: "prepareAfterSearch"}).then(() => {
    // Create a KeyboardEvent for pressing Enter.
    textarea.dispatchEvent(
        new KeyboardEvent('keydown', {
            key: "Enter",
            code: "Enter",
            keyCode: 13,
            which: 13,
            bubbles: true
        })
    );
    // Once we hit enter, a new page will be loaded.
    // So, it would be meaningless to write further code below.
});
```

This is the `scripts/afterSearch.js`:
```js
console.log("Search Done!");
```
