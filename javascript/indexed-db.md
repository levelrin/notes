## Glossary

|Word|Definition|Analogy in RDBs|
|--|--|--|
|Database|It contains multiple object stores.|Database|
|Object Store|It contains multiple records.|Table|
|Record|It's a JS object.|Row|

## Setup

```js
const dbName = "temp";
const dbVersion = 1;
const dbRequest = window.indexedDB.open(dbName, dbVersion);
// When the specified database doesn't exist, this event will be called.
// This can be used to prepare initial data.
dbRequest.onupgradeneeded = (event) => {
    const db = event.target.result;
    // We don't have to use `db.onerror` because it's for the event after db is fully opened.
    // If something fails in this phase, dbRequest.onerror will be triggered instead.
}
dbRequest.onsuccess = (event) => {
    const db = event.target.result;
    db.onerror = (error) => {
        console.error(`A database ${dbName} v${dbVersion} got an error: ${error}`);
    }
}
dbRequest.onerror = (event) => {
    console.error(`A database request for the ${dbName} v${dbVersion} got an error: ${event.target.error?.message}`);
}
dbRequest.onblocked = (event) => {
    console.error(`A database upgrade is blocked. The old version: ${event.oldVersion}, requested new version: ${event.newVersion}.`);
    // The db request is waiting until it's ready to upgrade.
    // That means refreshing the page after closing other tabs is not required in normal cases.
    alert("A database upgrade is needed, but other tabs with the same site are already open. Please close them.");
}
```
