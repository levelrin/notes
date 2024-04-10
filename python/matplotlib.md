## Plot Dots

```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4]
y = [0, 2, 4, 6]
fig, ax = plt.subplots()
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

## Combination of Dots and Lines

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

x = [1, 2, 3, 4]
y = [0, 2, 4, 6]
ax.scatter(x, y)

# Draw a straight line from (-1, 5) to (5, -1).
ax.plot([-1, 5], [5, -1], color="orange")

ax.scatter(0, 1, color="green")

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

# Set the display limits.
# We may use this to make some margins like this:
# ax.set_ylim(y_min - 0.5, y_max + 0.5)
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# Set the x and y grid units.
ax.set_xticks(np.arange(x_min, x_max + 1, x_grid_unit))
ax.set_yticks(np.arange(y_min, y_max + 1, y_grid_unit))

# Draw x and y axis to display origin.
ax.axhline(0, color="black", linewidth=1)
ax.axvline(0, color="black", linewidth=1)

# Make the length of x and y units the same.
ax.set_aspect("equal")

plt.show()
```

## Graph Function

```python
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

# Define a function like this.
def f(x): return 2 * x + 1

# It generates an array of 100 evenly spaced values from -10 to 10 (all-inclusive).
x_values = np.linspace(-10, 10, 100)
y_values = f(x_values)

plt.plot(x_values, y_values)

# Display the line in a grid.
x_min = -10
x_max = 10
y_min = -10
y_max = 10
x_grid_unit = 1
y_grid_unit = 1
ax.grid(True)
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xticks(np.arange(x_min, x_max + 1, x_grid_unit))
ax.set_yticks(np.arange(y_min, y_max + 1, y_grid_unit))
ax.axhline(0, color="black", linewidth=1)
ax.axvline(0, color="black", linewidth=1)
ax.set_aspect("equal")

plt.show()
```

## Labels

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.set_xlabel("Label for X-axis")
ax.set_ylabel("Label for Y-axis")
ax.set_title("Title")

x_0 = [0.5, 1, 1.5, 2, 2.5, 7.5, 8, 8.5, 9, 9.5]
y_0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
plt.scatter(x_0, y_0, color="red", label="Group 0")

x_1 = [4, 4.5, 5, 5.5, 6]
y_1 = [1, 1, 1, 1, 1]
plt.scatter(x_1, y_1, color="green", label="Group 1")

# Display labels for dots and lines.
# If the legend is in the way, we can change its position like this:
# plt.legend(bbox_to_anchor=(0.5, 0.6), loc="upper left")
# It means we put the upper left corner of legend at (0.5, 0.6) from the origin of the graph.
plt.legend()

plt.show()
```
