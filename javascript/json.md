## String to JSON

```js
const raw = `{"user1": {"name": "Rin"}}`;
const json = JSON.parse(raw);
console.log(json.user1.name);
```

## JSON to String

```js
const json = {user1: {name: "Rin"}}
const raw = JSON.stringify(json);
console.log(raw);
```
