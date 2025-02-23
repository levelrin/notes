## Cache Files

The `manifest.json`:
```json
{
  "name": "Hello PWA",
  "short_name": "HelloPWA",
  "start_url": "index.html",
  "display": "standalone",
  "icons": [
    {
      "src": "icons/android-chrome-512x512.png",
      "sizes": "512x512",
      "type": "image/png"
    },
    {
      "src": "icons/android-chrome-192x192.png",
      "sizes": "192x192",
      "type": "image/png"
    }
  ],
  "description": "PWA Sample App!",
  "screenshots": [
    {
      "src": "images/nugget-wide.png",
      "sizes": "1278x720",
      "type": "image/png",
      "form_factor": "wide",
      "label": "nugget"
    },
    {
      "src": "images/nugget-narrow.png",
      "sizes": "452x751",
      "type": "image/png",
      "form_factor": "narrow",
      "label": "nugget"
    }
  ]
}
```

The `index.html`:
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
<script type="text/javascript">
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker
            .register('/offline.js')
            .then(() => console.log('The service worker for the offline feature has been registered.'))
            .catch(error => console.log('Failed to register the service worker for the offline feature', error));
    } else {
        console.log('Using a service worker is unavailable.');
    }
</script>
</body>
</html>
```

The `offline.js`:
```js
const CACHE_NAME = 'hello-world-cache-v1';

// Cache all the files for offline usages.
self.addEventListener('install', event => {
    event.waitUntil(
        // The `caches` object is natively available by the browser.
        // Create a cache using a key.
        caches.open(CACHE_NAME).then(cache => {
            console.log('Cache files.')
            // Cache all the files used in the web app.
            // Behind the scenes, it's actually fetching (sending requests) using those paths and storing responses.
            // The origin (scheme + hostname + port) will be automatically determined by the browser.
            return cache.addAll([
                'manifest.json',
                'icons/android-chrome-512x512.png',
                'icons/android-chrome-192x192.png',
                'images/nugget-wide.png',
                'images/nugget-narrow.png',
                '/',
                '/index.html'
            ]).catch(error => {
                console.error('Failed to cache files.', error);
            });
        })
    );
});

// The `fetch` event is triggered whenever the app sends a network request.
self.addEventListener('fetch', event => {
    event.respondWith(
        // Check if the response to the request URL has been cached.
        // By the way, it won't match if the query parameters are different.
        // If we want to ignore the query parameters, we can use the { ignoreSearch: true } parameter like this: `caches.match(event.request, { ignoreSearch: true })`.
        caches.match(event.request).then(response => {
            if (response) {
                console.log('Serve the request from the cache. request: ' + event.request.url);
                // If a cached response exists (truthy), return it.
                return response;
            } else {
                console.log('Send an actual request: ' + event.request.url);
                // Otherwise (falsy), fetch from the network.
                return fetch(event.request);
            }
        })
    );
});

// Remove the old (no longer used) caches from the previous version.
self.addEventListener('activate', event => {
    event.waitUntil(
        caches.keys().then(keys => {
            return Promise.all(
                // Note that this removes all the caches except one with the matching cache name.
                // If there are multiple cache names, we need to modify this part.
                // By the way, caches are shared across the service workers if they are in the same scope.
                // Imagine there are two service worker files separately in the same scope (the same domain + path).
                // If they try to clean the up cache like this, they will end up deleting each other's cache.
                keys.filter(key => key !== CACHE_NAME)
                    .map(key => caches.delete(key))
            );
        })
    );
});
```

## Debugging

You may want to clear the cache for your PWA manually.

In a Chromium-based browser, you can do so by the following step: `Inspect` (F12) -> `Application` tab -> `Storage` section -> `Clear site data` button.
