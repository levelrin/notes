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

## Multiple Plots

The term `figure` represents a window.

The term `plot` represents a graph.

```python
import matplotlib.pyplot as plt
import numpy as np

# Create some data.
x = np.linspace(0, 2 * np.pi, 400)
y = np.sin(x ** 2)

# Create a figure and a 2x2 grid of subplots.
fig, axs = plt.subplots(2, 2)

# Adjust the vertical (hspace) and horizontal (wspace) spacing between subplots.
plt.subplots_adjust(hspace=0.4, wspace=0.4)

# Top left plot.
axs[0, 0].plot(x, y)
axs[0, 0].set_title('Axis [0, 0]')

# Top right plot.
# The third parameter is the format string.
# 'tab:orange' represents the orange color in the Tableau palette.
axs[0, 1].plot(x, y, 'tab:orange')
axs[0, 1].set_title('Axis [0, 1]')

# Bottom left plot.
axs[1, 0].plot(x, -y, 'tab:green')
axs[1, 0].set_title('Axis [1, 0]')

# Bottom right plot.
axs[1, 1].plot(x, -y, 'tab:red')
axs[1, 1].set_title('Axis [1, 1]')

plt.show()
```

## Animation

```python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def main():
    fig, ax = plt.subplots()

    # Prepare data.
    def f(x): return x ** 2
    x_values = np.linspace(-10, 10, 100)
    y_values = f(x_values)

    # Plot an empty data to get the line object.
    # We will use that line object to draw a real line with animation.
    # ax.plot([], []) returns a list with only one element (a Line2D object) in it.
    # As the name suggests, Line2D object is for drawing a line.
    # By the way, we may have multiple lines in the list depending on how many lines we plot.
    # For example, `ax.plot([], [], [], [])` will return 2 lines in the list.
    # The below line is the same as `line = ax.plot([], [])[0]`.
    # In Python, we can unpack the list and directly assign variables to each element using commas.
    line, = ax.plot([], [])

    # It updates the diagram on each frame (integer).
    # We can use the frame as the current index of the data list.
    def update(frame):
        # Plot the data until the current frame.
        line.set_data(x_values[:frame], y_values[:frame])
        # We must return the list of lines.
        return [line]

    # It's necessary to assign a variable.
    # The animation won't work without a variable because it will be garbage-collected... smh
    animation = FuncAnimation(
        fig=fig,
        # This function will be called on every frame.
        # It must have the frame parameter and return the list of `Artist` objects.
        # By the way, `Line2D` is a subtype of `Artist`.
        func=update,
        # Number of frames.
        frames=len(x_values),
        # Interval between each frame in milliseconds.
        # By the way, you may not be able to make it superfast by reducing it to a really low value
        # because the actual animation depends on the refreshment rate of your monitor.
        interval=50,
        # Do not restart the animation when it's done.
        repeat=False
    )

    # Fix the size of the diagram.
    x_min = -10
    x_max = 10
    y_min = 0
    y_max = 100
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    plt.show()

    # We can create an animation file like this.
    # By the way, the file is created in the current directory.
    animation.save("plot.gif")


main()
```
