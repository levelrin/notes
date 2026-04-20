Unfortunately, [GitHub Copilot](https://github.com/copilot) doesn't support [OpenAI API](https://openai.com/api/) out of the box.

For that reason, we need to run a proxy server (bridge server) to make Copilot available as an OpenAI API server.

We will use [Copilot OpenAI API](https://github.com/yuchanns/copilot-openai-api) for a bridge server.

Here is the `docker-compose.yml`:
```yml
services:
  copilot-bridge:
    image: ghcr.io/yuchanns/copilot-openai-api:v0.1.11
    container_name: copilot-bridge
    environment:
      # To get the token, you need to sign in to any native GitHub Copilot on your machine, such as the Copilot plugin on IntelliJ or VS Code.
      # Once you complete the authentication, you need to check the configuration file created by the plugin.
      # Usually, it's located at `~/.config/github-copilot/` in Linux and mac.
      # The token is written in `apps.json` or `hosts.json`.
      # Note that this token is different from the personal token, which starts with `ghp_`.
      # This token starts with `gho_`.
      - COPILOT_TOKEN=gho_...
      - COPILOT_SERVER_PORT=9191
    volumes:
      # You need to mount the Copilot's configuration folder.
      # Please check the comments for `COPILOT_TOKEN` above.
      - ~/.config/github-copilot/:/home/appuser/.config/github-copilot

  open-webui:
    image: ghcr.io/open-webui/open-webui:git-3e3f138-slim
    container_name: open-webui
    ports:
      - "3000:8080"
    environment:
      - OPENAI_API_BASE_URL=http://copilot-bridge:9191/v1
      # Same as the `COPILOT_TOKEN` above.
      - OPENAI_API_KEY=gho_...
    volumes:
      - ./open-webui:/app/backend/data
    depends_on:
      - copilot-bridge
```
