## Interface

We can specify which network interface to use like this:
```sh
tshark -i eth0
```

## Output Control

`tshark` will go through this process in order:
1. Capture Filter (-f)
2. Display Filter (-Y)
3. Output Formatting (-T -e)

### Capture Filter (-f)

It operates at a low level (raw packet headers).

For example, `tshark -i eth0 -f "tcp port 80"` will capture network traffic on port 80 only, which is suitable for capturing HTTP communication.

### Display Filter (-Y)

It operates at a high level.

For example, `tshark -i eth0 -f "tcp port 80" -Y "http"` will display the HTTP communication.

Note that only minimum information would be displayed.

To display more information, we need to specify the output format with the `-T` and `-e` flags.

### Output Formatting (-T -e)

We can specify the output format like this:
```sh
tshark -i eth0 -f "tcp port 80" -Y "http" -T fields -e http.request.full_uri -e http.request.line -e http.response_for.uri -e http.response.line
```

The `-T` flag determines the format.

The available formats are: `ek|fields|json|jsonraw|pdml|ps|psml|tabs|text`.

For example, if we want to display the output in JSON format, we can run the command like this:
```sh
tshark -i eth0 -f "tcp port 80" -Y "http" -T json -e http.request.full_uri -e http.request.line -e http.response_for.uri -e http.response.line
```

For the HTTP protocol, we can check the available fields from the [document](https://www.wireshark.org/docs/dfref/h/http.html).

Note that there is no field for general HTTP headers.

If we want to display general headers, we can use the verbose flag (-V) to display all (it can be tedious to parse, though).

Alternatively, we can use the fields `http.request.line` and `http.response.line` (it turns out to be headers, for some reason, which is confusing).

### Verbose

The following command will display every detail of HTTP communication because the verbose flag (-V) is used.
```sh
sudo tshark -i eth0 -Y "http" -V
```

## Save Filtered Output

We can capture the network traffic and save it to a file like this:
```sh
tshark -i eth0 -w traffic.pcap
```

However, we cannot use the display filter (-Y) for the `-w` flag because it's not supported.

Similarly, we cannot use the format flag (-T) either.

Thus, we need to write the console output into a file like this:
```sh
tshark -i eth0 -Y "http" -T json -V > traffic.json
```

This will create a file `traffic.json` and write the output into it.

If the file exists already, it will overwrite the output into it.
