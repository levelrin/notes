## URL

Let's say we have the following URL: `https://www.oursite.org:420/yoi?name=rin#section3`

The `origin` refers to `https://www.oursite.org:420`, a combination of `scheme`, `host`, and `port`.

The `authority` refers to `www.oursite.org:420`, a combination of `host` and `port`.

The `scheme` refers to `https`.

The `host` refers to `www.oursite.org`.

The `port` refers to `420`.

The `path` refers to `/yoi`.

The `query string` refers to `name=rin`.

The `fragment` refers to `section3`.

By the way, the URL may have the user information like this: `https://username:password@www.oursite.org:420/yoi?name=rin#section3`.

Note that the `origin` does not include the user information even though it's in between the `scheme` and the `host`.

Also, modern browsers typically ignore the user information in the URL for security reasons.

### Domain

The term `domain` refers to the registered name **within** the `host`.

Here is the summary of terminologies:

|Term|Example|Notes|
|---|---|---|
|Host|`www.oursite.org`|It's the whole thing. Subdomain + SLD + TLD|
|Subdomain|`www`|It's an optional part that extends the domain.|
|Domain|`oursite.org`|SLD + TLD|
|SLD|`oursite`|Second-Level Domain|
|TLD|`.org`|Top-Level Domain|
