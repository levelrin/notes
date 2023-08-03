## Pass arguments to the command

Example:
```sh
echo -ne '1\n1\n\n' | sh install.sh
```

`\n` represents `Enter` key press.

`-ne` is the combination of `-n` and `-e` flags.

`-n`: Do not output a trailing newline.

`-e`: Enable interpretation of the following backslash-escaped characters in each String, such as `\n`.
