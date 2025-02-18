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

## Richer Install UI

It's for providing screenshots and the app description on the installation screen.

Here is the sample configuration for it:
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

Note that the `description` and `screenshots` attributes are newly added.

The `"form_factor": "wide"` is for showing screenshots for the desktop.

Similarly, the `"form_factor": "narrow"` is for showing screenshots for the mobile.
