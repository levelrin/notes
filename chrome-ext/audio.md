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
    <title>Popup</title>
</head>
<body>
<button type="button" id="okay">Okay</button>
<button type="button" id="fahhh">Fahhh</button>
<script src="popup.js"></script>
</body>
</html>
```

---

`popup/popup.js`:
```js
window.onload = () => {
    const okay = document.getElementById("okay");
    okay.onclick = () => {
        // Message to `background.js`.
        chrome.runtime.sendMessage({
            about: "triggerAudio",
            audioId: "okay"
        });
    }
    const fahhh = document.getElementById("fahhh");
    fahhh.onclick = () => {
        chrome.runtime.sendMessage({
            about: "triggerAudio",
            audioId: "fahhh"
        });
    }
}
```

---

`background.js`:
```js
async function ensureOffscreen() {
    if (await chrome.offscreen.hasDocument()) {
        return;
    }
    // We can create only one offscreen.
    await chrome.offscreen.createDocument({
        url: "offscreen/offscreen.html",
        reasons: ["AUDIO_PLAYBACK"],
        justification: "To play audio."
    });
}
chrome.runtime.onMessage.addListener((message, __, ___) => {
    if (message.about === "triggerAudio") {
        ensureOffscreen().then(() => {
            // Message to `offscreen.js`.
            // Note that `chrome.runtime.sendMessage` is the broadcast (not only for the background process).
            chrome.runtime.sendMessage({
                about: "playAudio",
                audioId: message.audioId
            });
        });
    }
});
```

---

`offscreen/offscreen.html`:
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Offscreen</title>
</head>
<body>
<audio id="okay" src="/audio/okay.mp3"></audio>
<audio id="fahhh" src="/audio/fahhh.mp3"></audio>
<script src="offscreen.js"></script>
</body>
</html>
```

---

`offscreen/offscreen.js`:
```js
window.onload = () => {
    chrome.runtime.onMessage.addListener((message, __, ___) => {
        if (message.about === "playAudio") {
            const audio = document.getElementById(message.audioId);
            audio.pause();
            audio.currentTime = 0;
            audio.play();
        }
    });
}
```
