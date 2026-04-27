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
