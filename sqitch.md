## Access remote database

Please follow this structure:
```
db:pg://[user]:[password]@[host]:[port]/[db_name]
```

For example, see the below usages.

Deploy:
```sh
sqitch deploy db:pg://postgres:1111@localhost:5432/postgres --verify
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

Ex: `sqitch deploy --verify`

## Create a table

Add a sqitch plan:
```sh
sqitch add philosophers -n 'Add philosophers table.'
```

Modify `deploy/philosophers.sql` to:
```sql
BEGIN;

CREATE TABLE philosophers(
    name TEXT,
    school TEXT
);

COMMIT;
```

Modify `revert/philosophers.sql` to:
```sql
BEGIN;

DROP TABLE philosophers;

COMMIT;
```

Modify `verify/philosophers.sql` to:
```sql
BEGIN;

SELECT name, school
    FROM philosophers
WHERE FALSE;

ROLLBACK;
```

## Add rows

Add a sqitch plan:
```sh
sqitch add defaultPhilosophers -n 'Add default philosophers.'
```

Modify `deploy/defaultPhilosophers.sql` to:
```sql
BEGIN;

INSERT INTO philosophers (name, school) VALUES ('Mozi', 'Mohism');
INSERT INTO philosophers (name, school) VALUES ('Han Feizi', 'Legalism');

COMMIT;
```

Modify `revert/defaultPhilosophers.sql` to:
```sql
BEGIN;

DELETE FROM philosophers WHERE name='Mozi' OR name='Han Feizi';

COMMIT;
```

Modify `verify/defaultPhilosophers.sql` to:
```sql
BEGIN;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM philosophers WHERE name = 'Mozi')
       OR NOT EXISTS (SELECT 1 FROM philosophers WHERE name = 'Han Feizi') THEN
        RAISE EXCEPTION 'Either Mozi or Han Feizi (or both) not found in philosophers table';
    END IF;
END $$;

ROLLBACK;
```
