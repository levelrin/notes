## Using Docker

### Run postgres server

```sh
docker run --rm -d -p 5432:5432 -v $(pwd)/pgdata:/usr/share/pgdata/ --name pg -e POSTGRES_PASSWORD=1111 -e PGDATA=/usr/share/pgdata postgres:13.2
```

It will use the default username `postgres`.

### Access postgres container

```sh
docker container exec -it pg psql -U postgres
```

### Use the postgres client in my host machine to connect to the postgres server in Docker.

```sh
psql -h localhost -p 5432 -U postgres
```

## Dollar-Quoted String Constants

Since it's tedious to escape single quotes (') or backslashes (\\) when we write a string literals, Postgres offers another method called `dollar quoting`.

It has the following structure:
```
$<optional tag>$<string literals>$<optional tag>$
```

Here is an example:
```
$$Rin's note$$
$some tag$Rin's book$some tag$
```

Both yields the same string literal value: `Rin's note`.

The tag is case-sensitive.

refs: https://www.postgresql.org/docs/current/sql-syntax-lexical.html

## Function

We can define a function like this:
```sql
CREATE OR REPLACE FUNCTION sum(a INTEGER, b INTEGER) RETURNS INTEGER AS $$
BEGIN
    RETURN a + b;
END;
$$ LANGUAGE plpgsql;
```

Since we specified the language to `plpgsql`, we use `PL/pgSQL` (a procedural language for PostgreSQL).

The `BEGIN` and `END` keywords indicate that we use PL/pgSQL's block structure.

In that block, we can use PL/pgSQL's features, such as using the `RETURN` keyword.

Obviously, plain SQL does not have PL/pgSQL's features.

After executing the above SQL, we can check the new function by `\df` like this:
```
postgres=# \df
                           List of functions
 Schema |    Name     | Result data type |  Argument data types  | Type
--------+-------------+------------------+-----------------------+------
 public | sum         | integer          | a integer, b integer  | func
```

We can check if the function works fine like this:
```
postgres=# SELECT sum(3, 8);
 sum
-----
  11
(1 row)
```

## EXISTS

We can check if the table has a record like this:
```sql
SELECT EXISTS(SELECT 1 FROM philosophers WHERE name = 'Mozi');
```

A constant `1` is arbitrarily used.

It means the `SELECT` query will return 1 every time it finds a record that matches the `WHERE` clause.

We are using an arbitrary constant because we don't care about the result of the SELECT query.

All we need to know is whether a matching record exists or not.

The result may look like this:
```
 exists
--------
 t
(1 row)
```

## DO

The `DO` statement looks like this:
```plpgsql
DO $$ 
BEGIN 
    RAISE NOTICE 'Yoi Yoi'; 
END $$;
```

[This](https://www.postgresql.org/docs/current/sql-do.html) article says:
> `DO` executes an `anonymous code block`, or in other words a `transient anonymous function` in a procedural language.
>
> The code block is treated as though it were the body of a function with <ins>**no parameters, returning void**</ins>.

The `RAISE NOTICE` statement is used to print something.
