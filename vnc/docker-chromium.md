## About

It's about [docker-chromium](https://github.com/jlesage/docker-chromium).

## Quick Start

```sh
docker run --rm --name=chromium -p 5800:5800 -e ENABLE_CJK_FONT=1 -e WEB_AUDIO=1 -v $(pwd)/chromium:/config:rw ghcr.io/jlesage/chromium:v26.03.2
```

We can access the browser via http://localhost:5800.

`-e WEB_AUDIO=1` enables the audio.
Although the browser is now able to output audio (power given), it's off by default.
The user needs to click the hamburger menu on the left to expand the settings and turn on the audio and its volume.

`-e ENABLE_CJK_FONT=1"` installs East Asian fonts, such as Chinese, Japanese, and Korean.

Known Issue:
I was able to type English only.
The workaround is to copy the text from the host machine and paste it into Chromium.

## Remote Debugging Port

To enable the remote debugging port, we need to pass a parameter like `--remote-debugging-port=9223` when we run Chromium.

Unfortunately, `docker-chromium` doesn't offer a way to pass such a parameter at the time of writing.

Even if we pass the remote debugging port parameter, Chromium listens on the loopback interface (127.0.0.1) only for security reasons.

In other words, the remote debugging port is accessible only within the container, not from outside.

For that reason, a parameter like `--remote-debugging-address=0.0.0.0` didn't work.

Fortunately, we have workarounds for the above issues.

To pass unsupported parameters, we can overwrite the [params](https://github.com/jlesage/docker-chromium/blob/master/rootfs/etc/services.d/app/params) file, which `docker-chromium` uses to pass parameters to Chromium.

To resolve the loopback interface-only issue, we can expose the port using a relay tool such as [socat](http://www.dest-unreach.org/socat/).

To install `socat`, we can use the `INSTALL_PACKAGES` environment variable offered by `docker-chromium`.

Actually, we may also want to install [busybox-extras](https://pkgs.alpinelinux.org/package/edge/main/x86/busybox-extras) for [nc](https://linux.die.net/man/1/nc) to ensure we run `socat` when the remote debugging port is ready.

First, we create a file to overwrite the params:
```sh
vim chromium-params
```

The content should be like this:
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
# Note that we use port 9223, which is not the conventional one (9222).
printf "%s\n" "--remote-debugging-port=9223"

# Run socat when DevTools port is ready.
# Note that the text written into `stdout` will be used as parameters for Chromium.
# In other words, we should not print anything to `stdout` unless it's a parameter.
# Also, we need to run a logic on a separate process so that this script can finish.
if command -v socat >/dev/null && command -v nc >/dev/null; then
(
    while ! nc -z 127.0.0.1 9223; do sleep 1; done
    # Expose the DevTools port so that outsiders can access via port 9222.
    socat TCP-LISTEN:9222,fork TCP:127.0.0.1:9223
) >/dev/null 2>&1 &
fi
```

Next, we make it executable:
```sh
chmod +x chromium-params
```

Finally, we run the container like this:
```sh
docker run --rm --name=chromium -v $(pwd)/chromium:/config:rw -v $(pwd)/chromium-params:/etc/services.d/app/params:ro -p 5800:5800 -p 9222:9222 -e ENABLE_CJK_FONT=1 -e WEB_AUDIO=1 -e INSTALL_PACKAGES="socat busybox-extras" ghcr.io/jlesage/chromium:v26.03.2
```

Note that the following parameters are used for the remote debugging port:
 - `-v $(pwd)/chromium-params:/etc/services.d/app/params:ro`
 - `-p 9222:9222`
 - `-e INSTALL_PACKAGES="socat busybox-extras"`

We can confirm if the remote debugging port is working like this:
```sh
curl -v -k http://localhost:9222/json/version
```

You should get `200 OK`.
