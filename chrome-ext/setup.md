## Project Structure

```
project-name/
├── manifest.json
├── background.js
├── scripts/
│   └── content.js
├── popup/
│   ├── popup.html
│   ├── popup.js
│   └── popup.css
└── images/
    └── icons/
        ├── icon-16.png
        ├── icon-32.png
        ├── icon-48.png
        └── icon-128.png
```

## manifest.json

```json
{
   "manifest_version":3,
   "version":"0.1.0",
   "name":"Friendly Automator",
   "description":"Automate your tasks.",
   "icons":{
      "16":"images/icons/icon-16.png",
      "32":"images/icons/icon-32.png",
      "48":"images/icons/icon-48.png",
      "128":"images/icons/icon-128.png"
   },
   "action":{
      "default_popup":"popup/popup.html"
   }
}
```

## Popup

`popup.html`:
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Title</title>
    <link rel="stylesheet" href="popup.css">
</head>
<body>
<h1>Yoi Yoi</h1>
<script src="popup.js"></script>
</body>
</html>
```

`popup.js`:
```js
console.log("Yoi Yoi");
```

`popup.css`:
```css
h1 {
    color: red;
}
```

## Local Deployment

1. Go to the extension settings on your Chrome browser.
  Ex: chrome://extensions/
  Ex: brave://extensions/
2. Enable `Developer mode` located in the top right corner.
3. Click the `Load unpacked` button and select the project folder.

When you modify the code, click the refresh button on the extension settings page to reflect the change.
