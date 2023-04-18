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
