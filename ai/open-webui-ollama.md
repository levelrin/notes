## Quick Start

We can just run [Open WebUI](https://github.com/open-webui/open-webui) via Docker bundled with [Ollama](https://github.com/ollama/ollama) like this:
```sh
docker run --rm -d -p 3000:8080 --gpus=all -v "$(pwd)/ollama:/root/.ollama" -v "$(pwd)/open-webui:/app/backend/data" --name open-webui ghcr.io/open-webui/open-webui:ollama
```

Once the container is booted, we can access `http://localhost:3000`.

## Run Separately

We may want to run Ollama and Open WebUI separately.

For example, the bundled Ollama might not be up-to-date.

We can use `docker-compose.yml` like this:
```yml
services:
  ollama:
    image: ollama/ollama:latest
    container_name: ollama
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
    image: ghcr.io/open-webui/open-webui:main
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
