`manifest.json`:
```json
{
  "manifest_version":3,
  "version":"0.1.0",
  "name":"Test",
  "description":"Test",
  "icons":{
    "16":"images/icons/icon16.png",
    "32":"images/icons/icon32.png",
    "48":"images/icons/icon48.png",
    "128":"images/icons/icon128.png"
  },
  "action":{
    "default_popup":"popup/popup.html"
  },
  "permissions": ["offscreen"],
  "background": {
    "service_worker": "background.js"
  }
}
```

The important part is `"permissions": ["offscreen"]`.

---

`popup/popup.html`:
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Title</title>
</head>
<body>
<button type="button" id="button">Click Me!</button>
<script src="popup.js"></script>
</body>
</html>
```

---

`popup/popup.js`:
```js
window.onload = () => {
    const button = document.getElementById("button");
    button.onclick = () => {
        // Message to `background.js`.
        chrome.runtime.sendMessage({
            about: "triggerAudio"
        });
    }
}
```

---

`background.js`:
```js
chrome.runtime.onMessage.addListener((message, __, ___) => {
    if (message.about === "triggerAudio") {
        const audioUrl = "audio/audio.html";
        chrome.runtime.getContexts({
            contextTypes: ["OFFSCREEN_DOCUMENT"],
            documentUrls: [chrome.runtime.getURL(audioUrl)]
        }).then((contexts) => {
            if (contexts.length > 0) {
                // Message to `audio.js`.
                // Note that `chrome.runtime.sendMessage` is the broadcast (not only for the background process).
                chrome.runtime.sendMessage({
                    about: "playAudio"
                });
            } else {
                // Load `audio.html` if not already.
                chrome.offscreen.createDocument({
                    url: audioUrl,
                    reasons: ["AUDIO_PLAYBACK"],
                    justification: "To play audio."
                });
            }
        });
    }
});
```

---

`audio/audio.html`:
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Audio</title>
</head>
<body>
<audio controls id="okay" src="okay.mp3"></audio>
<script src="audio.js"></script>
</body>
</html>
```

The `okay.mp3` file is at `audio/okay.mp3`.

---

`audio/audio.js`:
```js
window.onload = () => {
    const okay = document.getElementById("okay");
    chrome.runtime.onMessage.addListener((message, __, ___) => {
        if (message.about === "playAudio") {
            // Play the audio from the message when the page has been loaded already.
            okay.play();
        }
    });
    // Play the audio when the page is loaded initially.
    okay.play();
}
```
