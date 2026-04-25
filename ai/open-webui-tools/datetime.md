## About

It's a tool for getting the datetime information.

## Code

```py
from datetime import datetime
from pydantic import BaseModel, Field


class Tools:
    def __init__(self):
        pass

    def current_datetime(self) -> str:
        """
        Get the current date and time from the system.
        :return: A string containing the current date and time.
        """
        try:
            now = datetime.now()
            formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
            return f"The current date and time is: {formatted_now}"
        except Exception as e:
            return f"Error retrieving datetime: {str(e)}"

```
