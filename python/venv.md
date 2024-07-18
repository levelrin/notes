## Initialize a virtual environment

```sh
python -m venv .venv
```

You may need to use `python3` instead of `python` command.

It will create the `.venv` directory.

## Activate the virtual environment

For Windows:
```sh
.venv\Scripts\activate
```

For Unix or MacOS:
```sh
source .venv/bin/activate
```

`(.venv)` will be displayed at the beginning of the terminal.

Now, we can install packages like this:
```sh
pip install requests
```

When we are done with the virtual environment, we can end it like this:
```sh
deactivate
```
