## About

We want to give an instruction (prompt) in Open WebUI to manipulate the browser.

In this example, we will use the Chromium browser and enable the remote debugging port.

Note that we will not use the headless mode. All the actions will be visible on screen for extra fun.

Since Open WebUI doesn't support stdio-based MCP servers, [MCPO](https://github.com/open-webui/mcpo), which is a tool that exposes any MCP tool as an OpenAPI-compatible HTTP server, is the go-to approach for using MCP servers.

To manipulate the browser via remote debugging port, we will use the [playwright](https://www.npmjs.com/package/@playwright/mcp?activeTab=readme) MCP server.

## Chromium

The setup is based on [this](https://github.com/levelrin/notes/blob/main/vnc/docker-chromium.md) investigation.

We create a file to enable the remote debugging port.
```sh
vim chromium-params
```

The content should be:
```sh
#!/bin/sh

set -u # Treat unset variables as an error.

if ! check_pid_namespace >/dev/null; then
    printf "%s\n" "--no-sandbox"
fi

printf "%s\n" "--disable-dev-shm-usage"
printf "%s\n" "--ignore-gpu-blocklist"
printf "%s\n" "--simulate-outdated-no-au='Tue, 31 Dec 2099 23:59:59 GMT'"
printf "%s\n" "--start-maximized"
printf "%s\n" "--user-data-dir=/config/chromium"

if [ -n "${CHROMIUM_APP_URL:-}" ]; then
    printf "%s\n" "--app=$CHROMIUM_APP_URL"
fi

# vim:ft=sh:ts=4:sw=4:et:sts=4


##### The code below is what we inject #####


# Enable remote debugging port.
printf "%s\n" "--remote-debugging-port=9222"
```

Since we will access the remote debugging port within the Chromium container, we don't need to expose it.

Let's make it executable:
```sh
chmod +x chromium-params
```

## Playwright

We create a configuration file:
```sh
vim playwright-config.json
```

The content should be:
```json
{
  "mcpServers": {
    "playwright": {
      "command": "npx",
      "args": ["-y", "@playwright/mcp@0.0.73", "--cdp-endpoint", "http://localhost:9222"]
    }
  }
}
```

Let's make it executable:
```sh
chmod +x playwright-config.json
```

## Docker Compose

Here is the `docker-compose.yml`:
```yml
services:
  chromium:
    image: ghcr.io/jlesage/chromium:v26.03.2
    container_name: chromium
    ports:
      # We can access the browser via http://localhost:5800
      - "5800:5800"
      # This is the port used by the MCP server (mcpo container).
      # We need to map the port here because mcpo is on Chromium's network (sidecar pattern).
      # We can check available functions provided by the MCP server via http://localhost:8000/docs
      - "8000:8000"
    environment:
      - ENABLE_CJK_FONT=1
      - WEB_AUDIO=1
    volumes:
      - ./chromium:/config:rw
      - ./chromium-params:/etc/services.d/app/params:ro
  mcpo:
    image: ghcr.io/open-webui/mcpo:git-788ff92
    container_name: mcpo
    # Since Chromium forces us to use localhost or an IP address for the remote debugging port, we put this container in Chromium's network.
    network_mode: "service:chromium"
    volumes:
      - ./playwright-config.json:/playwright-config.json
    # This is a general way to use an MCP server.
    # Any specific setup for the MCP server is done via a configuration file.
    command: --config /playwright-config.json
    depends_on:
      - chromium
  open-webui:
    image: ghcr.io/open-webui/open-webui:git-92dfa3f-cuda
    container_name: open-webui
    ports:
      - "3000:8080"
    volumes:
      - ./open-webui:/app/backend/data
    depends_on:
      - mcpo
```

Let's start the system:
```sh
docker compose up -d
```

We can confirm if the browser is up and running via http://localhost:5800

We can confirm if Open WebUI is up and running via http://localhost:3000

We can confirm if the MCP server is up and running via http://localhost:8000/docs

We can confirm if the remote debugging port is enabled like this:
```sh
docker exec mcpo curl -v -k http://localhost:9222/json/version
```

## Open WebUI

We can integrate the MCP server like this:
1. Go to `Settings`
2. Go to `Admin Settings`
3. Go to `Integrations`
4. In the `Manage Tool Servers` section, click the `Add Connection` button.
5. Put the `Name` and `Description`.
6. The URL should be: http://chromium:8000/playwright
7. Set `Auth` to None.
8. Click the `Verify Connection` button and `Save` if it was successful.

The MCP tools will be available in the model configuration.
