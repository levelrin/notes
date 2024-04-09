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

## Draw Lines

```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4]
y = [0, 2, 4, 6]
fig, ax = plt.subplots()
ax.plot(x, y)
plt.show()
```

## Draw Grid

```python
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.grid(True)

# All inclusive
x_min = -10
x_max = 10
y_min = -10
y_max = 10

x_grid_unit = 1
y_grid_unit = 1

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# Set the x and y grid units.
ax.set_xticks(np.arange(x_min, x_max + 1, x_grid_unit))
ax.set_yticks(np.arange(y_min, y_max + 1, y_grid_unit))

# Draw x and y axis to display origin.
ax.axhline(0, color='black', linewidth=1)
ax.axvline(0, color='black', linewidth=1)

# Make the length of x and y units the same.
ax.set_aspect('equal')

plt.show()
```
