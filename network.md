## Capture network traffic on the remote server

Run the command like this on the remote server:
```sh
tcpdump -i eth0 -w /tmp/traffic.pcap
```

`-i` to specify the network interface. Ex: `eth0`

`-w` to write the traffic on the specified file. Ex: `/tmp/traffic.pcap`

After capturing the traffic, we can get the `pcap` file from the remote to the local like this (it should be executed from the local terminal, not from the remote):
```sh
scp -l 1000 user@remote-domain-or-ip:/tmp/traffic.pcap ~/local/file/path/traffic.pcap
```

`-l 1000` is optional if you get errors due to large bandwidth. It limits bandwidth, specified in Kbit/s.

We can open the `pcap` file using Wireshark.

## Access the server within the same network

### Server running on Windows

Run the following Windows command while the server is running:
```sh
ipconfig
```

See the `IPv4 Address` of your network interface.
For example, if the Windows machine is connected to WiFi, you should look for the `Wireless LAN adapter Wi-Fi`.

Other devices should be able to access your server using that IP address as long as they are connected to the same WiFi.

### Server running on macOS

You can see your IP address like this: `System Settings` -> `Network` -> Select the connected network interface -> `Details...`.

Alternatively, you can run this command on Mac:
```sh
ifconfig | grep 'inet ' | grep -v '127.0.0.1'
```

`ifconfig` shows all the network interface information.

`grep 'inet '` means match the line containing the string 'inet '.

`grep -v '127.0.0.1' means exclude the line containing the string '127.0.0.1'.
We want to exclude the loopback address.

### Server running on WSL2 Docker

Open your WSL2 terminal and run this command to see your private IP for WSL2:
```sh
ifconfig | grep 'inet ' | grep -v '127.0.0.1'
```

Next, run Windows PowerShell as administrator and run the following commands.
(You need to replace `<port for your server>` and `<private IP for WSL2>`)
```command
netsh advfirewall firewall add rule name="Allowing LAN connections" dir=in action=allow protocol=TCP localport=<port for your server>
netsh interface portproxy add v4tov4 listenport=<port for your server> listenaddress=0.0.0.0 connectport=<port for your server> connectaddress=<private IP for WSL2>
```

The above command allows the firewall to accept inbound connections for the specified port and forward the port from the host to WSL2.

FYI, you can undo the above commands like this:
```command
netsh advfirewall firewall delete rule name="Allowing LAN connections"
netsh interface portproxy delete v4tov4 listenport=<port for your server> listenaddress=0.0.0.0
```

Now, you should be able to access the server from a different device using the server's host machine's private IP address.

Run the following Windows command to see the server's host machine's private IP:
```sh
ipconfig
```
