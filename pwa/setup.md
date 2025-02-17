## Bare Minimum Setup

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
</body>
</html>
```

`manifest.json`:
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
  ]
}
```

It's necessary to prepare icons.

The `manifest.json` is one of the crucial components that make the web app installable.

Furthermore, the web app has to be served via an HTTP server to be installable.

That means users probably need to reinstall the app if the domain changes.
