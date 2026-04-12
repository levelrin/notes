## Quick Start

We can just run [Open WebUI](https://github.com/open-webui/open-webui) via Docker bundled with [Ollama](https://github.com/ollama/ollama) like this:
```sh
docker run --rm -d -p 3000:8080 --gpus=all -v "$(pwd)/ollama:/root/.ollama" -v "$(pwd)/open-webui:/app/backend/data" --name open-webui ghcr.io/open-webui/open-webui:ollama
```

Once the container is booted, we can access `http://localhost:3000`.

## Run Separately

We may want to run Ollama and Open WebUI separately.

For example, the bundled Ollama might not be up-to-date.

### WSL

To enable the Nvidia GPU, the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) should be installed in WSL.

Please check the installation guide from [here](https://hub.docker.com/r/ollama/ollama).

After the installation, we can confirm by running the `nvidia-smi` command whether the GPU shows up.

We can use `docker-compose.yml` like this:
```yml
services:
  ollama:
    image: ollama/ollama:0.20.5
    container_name: ollama
    tty: true
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    volumes:
      - ./ollama:/root/.ollama

  open-webui:
    image: ghcr.io/open-webui/open-webui:git-92dfa3f-cuda
    container_name: open-webui
    ports:
      - "3000:8080"
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
    volumes:
      - ./open-webui:/app/backend/data
    depends_on:
      - ollama
```

To confirm the GPU usage, do the following:
1. Initiate a chat, which will initiate the model. Make sure the model has finished replying.
2. Get into the Ollama container: `docker exec -it ollama /bin/bash`.
3. Run `ollama ps`. The `PROCESSOR` column should show the GPU usage.

### Apple Silicon

Unfortunately, Docker on macOS does not support GPU passthrough for Apple Silicon (Metal).

To enable the GPU, we have to run Ollama as a native macOS application rather than inside a container.

We can at least run Open WebUI using Docker Compose like this:
```yml
services:
  open-webui:
    image: ghcr.io/open-webui/open-webui:git-92dfa3f
    container_name: open-webui
    ports:
      - "3000:8080"
    environment:
      - OLLAMA_BASE_URL=http://host.docker.internal:11434
    extra_hosts:
      - "host.docker.internal:host-gateway"
    volumes:
      - ./open-webui:/app/backend/data
    restart: always
```
