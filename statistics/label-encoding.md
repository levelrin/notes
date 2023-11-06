It's a way to convert discrete variables into numbers.

Let's say we have a data like this:

| Food     | Calories |
| -------- | -------- |
| Apple    | 95       |
| Chicken  | 231      |
| Broccoli | 50       |

The discrete variables (food) can be converted into this using label encoding:

| Food     | Calories |
| -------- | -------- |
| 0        | 95       |
| 1        | 231      |
| 2        | 50       |

0 represents Apple,<br>
1 represents Chicken,<br>
2 represents Broccoli

---

A drawback of label encoding is that the numbers are arbitrarily chosen.

It's not a good idea to use label encoding if the algorithm expects some meaning to the magnitude of converted numbers.

For example, the algorithm might be biased towards higher values.
