## Start a new container and be ready to execute commands in there

```sh
docker run --rm --name c1 -it ubuntu:23.04 /bin/bash
```

## ssh into a container

```sh
docker container exec -it c1 /bin/bash
```

## Expose port on a running container

Docker does not support exposing the port on a running container at the time of writing.

It's only possible when we run a container.

For a workaround, we can create an image of the running container, stop it, and rerun it.

#### Create an image of the running container

```sh
docker commit c1 temp:1
```

#### Expose port when we run a container

```sh
docker run --rm --name c1 -p 4321:4321 -it temp:1 /bin/bash
```

#### Remove the image after using the container

```sh
docker rmi temp:1
```

## Communicate between containers in the same network

#### Create a new network

```sh
docker network create n1
```

#### Run a container in the network

```sh
docker run --rm --name c1 --network n1 -it temp:1 /bin/bash
```

#### Check the status of the network

```sh
docker inspect n1
```

#### Send a request from one container to another in the same network

Let's say the container `c1` and `c2` are in the same network `n1`.

And `c1` is the web server listening to the port 4321, and `c2` is the client.

The container `c2` can use `c1` as a hostname to send a request like this:
```sh
curl http://c1:4321
```
