## Glossary

|Word|Definition|Analogy in RDBs|
|--|--|--|
|Database| Contains multiple object stores.|Database|
|Object Store|Contains multiple records.|Table|
|Record|A single JavaScript object stored in an object store.|Row|
|Key|A unique identifier for each record. Always unique within the object store.|Primary Key|
|Index|We can set a record's property as an index. We can query a record using the index instead of the key. For example, we can find a user using their name (index) instead of their ID (key). Can be unique or non-unique.|Index|

## Setup

```js
const dbName = "temp";
const dbVersion = 1;
const dbRequest = window.indexedDB.open(dbName, dbVersion);
// When the specified database doesn't exist, this event will be called.
// This can be used to prepare initial data.
dbRequest.onupgradeneeded = (event) => {
    // We don't have to use `db.onerror` because it's for the event after db is fully opened.
    // If something fails in this phase, dbRequest.onerror will be triggered instead.
    const db = event.target.result;
    // The object store can be created only during the upgrade because it involves the schema changes.
    const store = db.createObjectStore(
        "users",
        // Optional parameters:
        {
            // It's like a primary key.
            // If it's not specified here, we must specify the key every time we add or put a record later.
            keyPath: "id",
            //Every time we add an item, it increments the primary key automatically.
            // Actually, the key doesn't necessarily have to be a number.
            // For example, we can use strings for the key.
            // In that case, we cannot set `autoIncrement` to true. It will be ignored anyway.
            autoIncrement: true
        }
    );
    // If we are going to query data based on the "name" instead of the primary key "id", it's better to create an index like this for faster lookup.
    // The first parameter is the name of the index. It can be anything like "byName" or "name".
    // The second parameter is the actual property of the object. For example, we can search the user(s) by looking at the `user.name` property.
    store.createIndex("name", "name", { unique: false });
    store.createIndex("email", "email", { unique: true });
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

## Define Schema

```js
const dbName = "temp";
const dbVersion = 1;
const dbRequest = window.indexedDB.open(dbName, dbVersion);
dbRequest.onupgradeneeded = (event) => {
    const db = event.target.result;
    // The object store can be created only during the upgrade because it involves the schema changes.
    const store = db.createObjectStore(
        "users",
        // Optional parameters:
        {
            // It's like a primary key.
            // If it's not specified here, we must specify the key every time we add or put a record later.
            keyPath: "id",
            //Every time we add an item, it increments the primary key automatically.
            // Actually, the key doesn't necessarily have to be a number.
            // For example, we can use strings for the key.
            // In that case, we cannot set `autoIncrement` to true. It will be ignored anyway.
            autoIncrement: true
        }
    );
    // If we are going to query data based on the "name" instead of the primary key "id", it's better to create an index like this for faster lookup.
    // The first parameter is the name of the index. It can be anything like "byName" or "name".
    // The second parameter is the actual property of the object. For example, we can search the user(s) by looking at the `user.name` property.
    store.createIndex("name", "name", { unique: false });
    store.createIndex("email", "email", { unique: true });
}
```

## Prepare Initial Data

```js
const dbName = "temp";
const dbVersion = 1;
const dbRequest = window.indexedDB.open(dbName, dbVersion);
dbRequest.onupgradeneeded = (event) => {
    const db = event.target.result;
    const store = db.createObjectStore("users", { keyPath: "id", autoIncrement: true });
    store.createIndex("name", "name", { unique: false });
    store.createIndex("email", "email", { unique: true });
    // We shouldn't use `db.transaction` because the db request is not finished yet.
    // Instead, we can use the store we just created.
    // Any error would be caught by the dbRequest.onerror.
    store.add({
        name: "Rin",
        email: "levelrin@gmail.com"
    });
}
dbRequest.onerror = (event) => {
    console.error(`A database request for the ${dbName} v${dbVersion} got an error: ${event.target.error?.message}`);
}
```

## Retrieve Data

### Get Multiple Values

```js
const dbName = "temp";
const dbVersion = 1;
const dbRequest = window.indexedDB.open(dbName, dbVersion);
dbRequest.onupgradeneeded = (event) => {
    const db = event.target.result;
    const store = db.createObjectStore("users", { keyPath: "id", autoIncrement: true });
    store.createIndex("name", "name", { unique: false });
    store.createIndex("email", "email", { unique: true });
    store.add({ name: "Rin", email: "levelrin@gmail.com" });
}
dbRequest.onsuccess = (event) => {
    const db = event.target.result;
    db.onerror = (error) => {
        console.error(`A database ${dbName} v${dbVersion} got an error: ${error}`);
    }
    // Here is the flow: database -> transaction -> object store -> request.
    // In this case, it seems the parameter "users" is duplicated.
    // However, the first part defines the scope of the transaction.
    // That means we can actually have multiple object store names like this:
    // const transaction = db.transaction(["users", "devices"], "readonly");
    // And then, we can select the object store like this:
    // const store = transaction.objectStore("devices");
    const store = db.transaction("users", "readonly").objectStore("users");
    const getAllRequest = store.getAll();
    getAllRequest.onsuccess = () => {
        const users = getAllRequest.result;
        // Output: [{"name":"Rin","email":"levelrin@gmail.com","id":1}]
        console.log("All users:", JSON.stringify(users));
    };
    getAllRequest.onerror = (error) => {
        console.error("Failed to fetch users:", error.target.error);
    };
}
```
