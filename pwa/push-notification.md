## Disclaimer

I realized that the app's behavior depends on browsers.

I served my app via http://localhost:7070.
I used Chrome on Windows, which worked the best.

The push notification subscription was aborted on the Brave browser.
It seems the [push API](https://developer.mozilla.org/en-US/docs/Web/API/Push_API) is blocked by default on Brave unless the user configures it to allow it.

Also, I noticed that the push notification permission dialog is not displayed on Edge.
I had to manually allow it by clicking the icon next to the URL.

Apparently, for non-localhost, the app must be served by HTTPS (I haven't confirmed).

## Send Simple Push Notification

First, we need to create a [VAPID](https://datatracker.ietf.org/doc/html/rfc8292) key pair.

We can easily generate it using the [web-push](https://www.npmjs.com/package/web-push) command-line tool.

Here's how to install using [npm](https://www.npmjs.com/):
```sh
npm install web-push -g
```

The `-g` flag makes the `web-push` package a global command-line tool.

And then, we use this command to generate a key pair:
```sh
web-push generate-vapid-keys
```

The output would look like this:
```
=======================================

Public Key:
BNSY-HyDDmyP13Nq2atA...

Private Key:
C0qSnPSHp0f_bLIJ...

=======================================
```

Now, we are ready to write some code.

`index.html`:
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hello PWA</title>
    <link rel="manifest" href="manifest.json">
</head>
<body>
<h1>Hello, PWA!</h1>
<script type="text/javascript" src="index.js"></script>
</body>
</html>
```

`index.js`:
```js
if ("serviceWorker" in navigator) {
    navigator.serviceWorker
        .register("web-push.js")
        .then(() => console.log("The service worker for the push notifications feature has been registered."))
        .catch(error => console.error("Failed to register the service worker for the push notifications feature.", error));
    navigator.serviceWorker.ready.then((registration) => {
        return registration.pushManager.getSubscription().then((subscription) => {
            // If a subscription was found, return it.
            if (subscription) {
                return subscription;
            }
            const vapidPublicKey = "BNSY-HyDDmyP13Nq2atA...";
            return registration.pushManager.subscribe({
                userVisibleOnly: true,
                applicationServerKey: urlBase64ToUint8Array(vapidPublicKey)
            });
        });
    }).then((subscription) => {
        console.log("Subscription JSON:", JSON.stringify(subscription));
    });
    // Print the payload of the push notification.
    navigator.serviceWorker.addEventListener("message", (event) => {
        if (event.data && event.data.type === "PUSH_NOTIFICATION") {
            const jsonPayload = event.data.data;
            console.log("Received a push notification. title: " + jsonPayload.title + ", body: " + jsonPayload.body);
        }
    });
} else {
    console.log('Using a service worker is unavailable.');
}

/**
 * This code is based on https://github.com/mdn/serviceworker-cookbook/blob/master/tools.js
 *
 * @param base64String A VAPID public key.
 * @returns {Uint8Array<ArrayBuffer>} It's for the parameter `applicationServerKey` when we subscribe.
 */
function urlBase64ToUint8Array(base64String) {
    const padding = '='.repeat((4 - base64String.length % 4) % 4);
    const base64 = (base64String + padding)
        .replace(/\-/g, '+')
        .replace(/_/g, '/');
    const rawData = window.atob(base64);
    const outputArray = new Uint8Array(rawData.length);
    for (let index = 0; index < rawData.length; ++index) {
        outputArray[index] = rawData.charCodeAt(index);
    }
    return outputArray;
}
```

`web-push.js` (responsible for handling the push notification, such as displaying the push notification to the user):
```js
self.addEventListener("push", (event) => {
    console.log("Push event received.", event);
    // Retrieve the textual payload from event.data (a PushMessageData object).
    // Other formats are supported (ArrayBuffer, Blob, JSON), check out the documentation
    // on https://developer.mozilla.org/en-US/docs/Web/API/PushMessageData.
    const payload = event.data ? event.data.text() : "no payload";
    console.log("Push event payload:", payload);
    const jsonPayload = JSON.parse(payload);
    // Keep the service worker alive until the notification is created.
    event.waitUntil(
        self.clients.matchAll({ includeUncontrolled: true }).then((allClients) => {
            // Send the push notification content to the web apps so they can show it.
            allClients.forEach((client) => {
                client.postMessage({
                    type: "PUSH_NOTIFICATION",
                    data: jsonPayload
                });
            });
        }).finally(() => {
            // Always show the notification, even if no clients are open.
            return self.registration.showNotification(
                jsonPayload.title,
                { body: jsonPayload.body }
            );
        })
    );
});
```

Now, we can deploy the app, allow the notification (hopefully from the dialog), and check the console.

If everything went well, we will see an output like this:
```
Subscription JSON: {"endpoint":"https://fcm.googleapis.com/fcm/send/fhslqvtyni8:APA91bHLv...","expirationTime":null,"keys":{"p256dh":"BEjuX6t4x...","auth":"s0tC..."}}
```

In other words, this is how a subscription JSON object looks like:
```json
{
   "endpoint":"https://fcm.googleapis.com/fcm/send/fhslqvtyni8:APA91bHLv...",
   "expirationTime":null,
   "keys":{
      "p256dh":"BEjuX6t4x...",
      "auth":"s0tC..."
   }
}
```

Obviously, we need to use that subscription object to send a push notification.

The easiest way is to use the `web-push` command-line tool again, like this:
```sh
web-push send-notification \
  --endpoint="https://fcm.googleapis.com/fcm/send/fhslqvtyni8:APA91bHLv..." \
  --key="BEjuX6t4x..." \
  --auth="s0tC..." \
  --vapid-subject="mailto:levelrin@gmail.com" \
  --vapid-pubkey="BNSY-HyDDmyP13Nq2atA..." \
  --vapid-pvtkey="C0qSnPSHp0f_bLIJ..." \
  --payload='{"title":"Hello","body":"Yoi Yoi"}'
```

The format of the `--payload` parameter can be anything (defined by the application).
