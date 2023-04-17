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

Since it's tedious to specify the remote, we can modify `sqitch.conf` like this:
```
[core]
        engine = pg
        # plan_file = sqitch.plan
        # top_dir = .
[engine "pg"]
        target = db:pg://postgres:1111@localhost:5432/postgres
        # registry = sqitch
        # client = /usr/bin/psql
```

We no longer need to specify the remote.
