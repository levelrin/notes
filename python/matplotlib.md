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

## Text

```python
import matplotlib.pyplot as plt


def main():
    x = [1, 2, 3, 4, 5]
    y = [2, 3, 5, 7, 11]
    plt.scatter(x, y)
    # Write a text to the specified coordinate.
    # `horizontalalignment=left` places the text on the right side of the dot.
    # `verticalalignment=bottom` places the text on the top of the dot.
    plt.text(3, 5, "Yoi", horizontalalignment="left", verticalalignment="bottom", weight="semibold")
    plt.show()


main()
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
# `axs` is usually a 2D array except for the `plt.subplots(1, 2)` or `plt.subplots(2, 1)`.
# In those cases, `axs` becomes 1D array, so we need to plot by `axs[0]` or `axs[1]`.
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

## Draw 3D Lines

```python
import matplotlib.pyplot as plt
import numpy as np


def main():
    x_values = np.linspace(0, 10, 100)
    y_values = np.linspace(0, 10, 100)
    z_values = np.linspace(0, 10, 100)
    ax = plt.axes(projection="3d")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.plot3D(x_values, y_values, z_values)
    plt.show()


main()
```

## Draw 3D Surface

```python
import matplotlib.pyplot as plt
import numpy as np


def main():
    ax = plt.axes(projection="3d")

    # We must use numpy arrays because `ax.plot_surface` depends on it.
    # Furthermore, we must use 2D arrays.
    # In this case, we use 4 coordinates to draw a plane.
    # In other words, (0, 0, 5), (5, 0, 5), (0, 5, 5), and (5, 5, 5) are used.
    # You may wonder why we have to use 2D arrays.
    # When it comes to drawing a surface in a 3D space, we can think of it as having a sheet of paper
    # and bending it or just keeping it flat and placing it at some angle.
    # To make a flat paper (grid), we need to duplicate x values by the number of y values.
    # For example, we have x_values = [1, 2, 3].
    # If we want to make a paper (grid) with height=2, we need to duplicate x_values 2 times.
    # So, we use 2D arrays to represent grids, in which the number of outer arrays corresponds to the number of y_values,
    # and the number of inner arrays corresponds to the number of x_values.
    x_values = np.array([[0, 5], [0, 5]])
    y_values = np.array([[0, 0], [5, 5]])
    z_values = np.array([[5, 5], [5, 5]])
    # `alpha` controls the opacity. Range: [0, 1].
    ax.plot_surface(x_values, y_values, z_values, color="blue", alpha=0.5)

    # The following is a more common approach to drawing a surface.
    second_x_values = np.linspace(0, 5, 10)
    second_z_values = np.linspace(2.5, 7.5, 10)
    # As we know already, we have to use 2D arrays.
    # `np.meshgrid` conveniently create 2D arrays in a grid pattern for two axis (x and z in this case).
    # For example, we have x = [1, 2, 3, 4] and y = [5, 6, 7].
    # To make a flat surface (grid), we need coordinates like below:
    # ```
    # (1, 7) (2, 7) (3, 7) (4, 7)
    # (1, 6) (2, 6) (3, 6) (4, 6)
    # (1, 5) (2, 5) (3, 5) (4, 5)
    # ```
    # That means we need x = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
    # and y = [[7, 7, 7, 7], [6, 6, 6, 6], [5, 5, 5, 5]].
    # So, `np.meshgrid` returns the first and second 2D arrays that we needed.
    x_mesh, z_mesh = np.meshgrid(second_x_values, second_z_values)
    # Since we want y values to be constant, we use `np.full_like` to make a 2D array with constant values.
    second_y_values = np.full_like(x_mesh, 2.5)
    ax.plot_surface(x_mesh, second_y_values, z_mesh, color="red", alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


main()
```
