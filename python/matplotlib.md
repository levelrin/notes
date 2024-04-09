## Plot Dots

```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4]
y = [0, 2, 4, 6]
fig, ax = plt.subplots()
ax.set_xlabel("label x")
ax.set_ylabel("label y")
ax.set_title("Title")
ax.scatter(x, y)
plt.show()
```
