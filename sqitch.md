## Access remote database

Please follow this structure:
```
db:pg://[user]:[password]@[host]:[port]/[db_name]
```

For example, see the below usages.

Deploy:
```sh
sqitch deploy db:pg://postgres:1111@localhost:5432/postgres
```

Revert:
```sh
sqitch revert db:pg://postgres:1111@localhost:5432/postgres
```

Verify:
```sh
sqitch verify db:pg://postgres:1111@localhost:5432/postgres
```
