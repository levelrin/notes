## Open Internal Page On New Tab

We want to open the `popup/settings.html` on a new tab.

Here is the `manifest.json`:
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
  "permissions": ["tabs"]
}
```

This is the relevant part:
```
"permissions": ["tabs"]
```

Here is the `popup/popup.html`:
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Title</title>
    <link rel="stylesheet" href="popup.css">
</head>
<body>
<!-- Note that using inline script is not allowed in Chrome extension due to Content Security Policy (CSP).-->
<!-- For example, the following inline JS `openSettingsPage()` won't be executed: -->
<!-- <button type="button" onclick="openSettingsPage()">Settings</button> -->
<button type="button" id="settingsButton">Settings</button>
<script src="popup.js"></script>
</body>
</html>
```

Here is the `popup/popup.js`:
```js
document.addEventListener('DOMContentLoaded', () => {
    const settingsButton = document.getElementById('settingsButton');
    settingsButton.addEventListener('click', () => {
        chrome.tabs.create({
            url: chrome.runtime.getURL('popup/settings.html')
        });
    });
});
```

Here is the `popup/settings.html`:
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Settings</title>
</head>
<body>
<h1>Settings</h1>
</body>
</html>
```
