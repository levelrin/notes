It's a way to convert discrete variables into numbers.

Let's say we have a data like this:

| Food     | Calories |
| -------- | -------- |
| Apple    | 95       |
| Chicken  | 231      |
| Broccoli | 50       |

The discrete variables (food) can be converted into this using one-hot encoding:

| Apple | Chicken | Broccoli | Calories |
| ----- | ------- | -------- | -------- |
| 1     | 0       | 0        | 95       |
| 0     | 1       | 0        | 231      |
| 0     | 0       | 1        | 50       |

A drawback of one-hot encoding is that we will have to create many new columns if we have many discrete variables.
