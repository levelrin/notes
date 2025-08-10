## Interface

We can specify which network interface to use like this:
```sh
tshark -i eth0
```

We can use `any` to select all interfaces:
```sh
tshark -i any
```

However, it may not work well on Mac.

For example, we noticed that only the HTTP request was captured, and the response was not captured on Mac.

Perhaps the following message from the command was related:
> tshark: BIOCPROMISC: Operation not supported on socket.

## Output Control

`tshark` will go through this process in order:
1. Capture Filter (-f)
2. Display Filter (-Y)
3. Output Formatting (-T -e)

### Capture Filter (-f)

It operates at a low level (raw packet headers).

For example, `tshark -i any -f "tcp port 80"` will capture network traffic on port 80 only, which is suitable for capturing HTTP communication.

### Display Filter (-Y)

It operates at a high level.

For example, `tshark -i any -f "tcp port 80" -Y "http"` will display the HTTP communication.

Note that only minimum information would be displayed.

To display more information, we need to specify the output format with the `-T` and `-e` flags.

### Output Formatting (-T -e)

We can specify the output format like this:
```sh
tshark -i any -f "tcp port 80" -Y "http" -T fields -e http.request.full_uri -e http.request.line -e http.response_for.uri -e http.response.line
```

The `-T` flag determines the format.

The available formats are: `ek|fields|json|jsonraw|pdml|ps|psml|tabs|text`.

For example, if we want to display the output in JSON format, we can run the command like this:
```sh
tshark -i any -f "tcp port 80" -Y "http" -T json -e http.request.full_uri -e http.request.line -e http.response_for.uri -e http.response.line
```

For the HTTP protocol, we can check the available fields [here](https://www.wireshark.org/docs/dfref/h/http.html).

Note that there is no field for general HTTP headers.

If we want to display general headers, we can use the verbose flag (-V) to display all (it can be tedious to parse, though).

Alternatively, we can use the fields `http.request.line` and `http.response.line` (it turns out to be headers, for some reason, which is confusing).

For the HTTP 2 protocol, we can check the available fields [here](https://www.wireshark.org/docs/dfref/h/http2.html).

Note that the fields are quite different from HTTP 1, for some reason.

Also, we need to use the filter `-Y "http2"`.

### Verbose

The following command will display every detail of HTTP communication because the verbose flag (-V) is used.
```sh
tshark -i any -Y "http" -V
```

## Save Filtered Output

We can capture the network traffic and save it to a file like this:
```sh
tshark -i any -w raw-traffic.pcapng
```

However, we cannot use the display filter (-Y) for the `-w` flag because it's not supported.

Similarly, we cannot use the format flag (-T) either.

Thus, we need to write the console output into a file like this:
```sh
tshark -i any -Y "http" -T json -V > traffic.json
```

This will create a file `traffic.json` and write the output into it.

If the file exists already, it will overwrite the output into it.

## Decrypt TLS

We can easily decrypt TLS if the client supports [SSLKEYLOGFILE](https://www.ietf.org/archive/id/draft-thomson-tls-keylogfile-00.html).

`SSLKEYLOGFILE` is the environment variable that we need to set like this:
```sh
export SSLKEYLOGFILE=/home/rin/temp/tls-keys.log
```

When the client starts TLS communication, it appends the TLS session key to the file that we specified via the `SSLKEYLOGFILE`.

Although many clients, such as chromium-based browsers, recognize the practice of using the `SSLKEYLOGFILE` environment variable, we should check if the client supports it by checking if `SSLKEYLOGFILE` is populated.

Once the `SSLKEYLOGFILE` is ready, we can decrypt the communication like this:
```sh
tshark -i any -o tls.keylog_file:/home/rin/temp/tls-keys.log -Y "http2" -T json -e http2.header.name -e http2.header.value -e http2.headers.status -e http2.body.reassembled.data
```

## Check which protocols are used

We can use the `-z io,phs` flag like this:
```sh
tshark -r raw-traffic.pcapng -o tls.keylog_file:/home/rin/temp/tls-keys.log -z io,phs
```

The `-z` flag is for generating the statistics.

The `io` stands for Input/Output.

The `phs` stands for Protocol Hierarchy Statistics.

`io` and `phs` always go together (like this: `io,phs`) by design.

We can think of `phs` as an inner category of `io`.
