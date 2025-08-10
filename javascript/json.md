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

## Do something only when the parse was successful

```js
const json = {school: {"subject.economics": {macro: "Keynesian"}}}
const macro = json.school?.["subject.economics"]?.macro;
if (macro !== undefined) {
    console.log("Macro: ", macro);
}
const labor = json.school?.["subject.economics"]?.labor;
if (labor !== undefined) {
    console.log("Labor: ", labor);
}
```

## Check if JSON array has the item

```js
const raw = `[":method", ":scheme", ":authority", ":path", "user-agent", "accept"]`;
const json = JSON.parse(raw);
if (json.includes(":authority")) {
    console.log("Found!");
}
```
