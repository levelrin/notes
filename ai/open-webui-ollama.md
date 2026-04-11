## Quick Start

We can just run [Open WebUI](https://github.com/open-webui/open-webui) via Docker bundled with [Ollama](https://github.com/ollama/ollama) like this:
```sh
docker run --rm -d -p 3000:8080 --gpus=all -v "$(pwd)/ollama:/root/.ollama" -v "$(pwd)/open-webui:/app/backend/data" --name open-webui ghcr.io/open-webui/open-webui:ollama
```

Once the container is booted, we can access `http://localhost:3000`.
