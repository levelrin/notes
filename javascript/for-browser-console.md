## Automatic Scroll

Start scrolling down every second:
```javascript
const loop = setInterval(() => {window.scrollBy(0, 1000);}, 1000);
```

Stop scrolling:
```javascript
clearInterval(loop);
```
