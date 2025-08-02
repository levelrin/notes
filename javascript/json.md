## String to JSON

```js
const raw = `{"school": {"subject.economics": {"macro": "Keynesian"}}}`;
const json = JSON.parse(raw);
console.log(json.school["subject.economics"].macro);
```

## JSON to String

```js
const json = {school: {"subject.economics": {macro: "Keynesian"}}}
const raw = JSON.stringify(json);
console.log(raw);
```
