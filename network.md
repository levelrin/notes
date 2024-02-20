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
