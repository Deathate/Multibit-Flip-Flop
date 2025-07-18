# %%
import os

import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image


# Quartic bump function and its gradient
def bump_1d(u):
    return (1 - u**2) ** 2 if abs(u) <= 1 else 0


def bump_grad_1d(u):
    if abs(u) > 1:
        return 0
    return -4 * u * (1 - u**2)


def bump_2d(dx, dy):
    return bump_1d(dx) * bump_1d(dy)


def bump_grad_2d(dx, dy):
    dphi_dx = bump_grad_1d(dx) * bump_1d(dy)
    dphi_dy = bump_grad_1d(dy) * bump_1d(dx)
    return dphi_dx, dphi_dy


# Configuration
grid_size = (50, 50)
bin_width = 1.0
target_density = 1.0
density_weight = 1.0
lr = 0.1
steps = 30

# Initial cell positions
initial_cells = [
    {"x": 20.0, "y": 25.0, "w": 4.0, "h": 4.0},
    {"x": 23.0, "y": 25.0, "w": 4.0, "h": 4.0},
    # {"x": 23.0, "y": 25.0, "w": 8.0, "h": 4.0},
]


# Compute density and gradients
def compute_density_and_gradients(cells):
    density_grid = np.zeros(grid_size)
    gradients = [[0.0, 0.0] for _ in cells]

    for idx, cell in enumerate(cells):
        x, y, w, h = cell["x"], cell["y"], cell["w"], cell["h"]
        rx, ry = w / 2, h / 2
        area = w * h

        x_start = int(max(0, (x - rx) // bin_width))
        x_end = int(min(grid_size[0], (x + rx) // bin_width + 1))
        y_start = int(max(0, (y - ry) // bin_width))
        y_end = int(min(grid_size[1], (y + ry) // bin_width + 1))

        for i in range(x_start, x_end):
            for j in range(y_start, y_end):
                bin_x = i * bin_width + 0.5
                bin_y = j * bin_width + 0.5

                dx = (bin_x - x) / rx
                dy = (bin_y - y) / ry

                phi = bump_2d(dx, dy)
                contrib = area * phi
                density_grid[i, j] += contrib

    for idx, cell in enumerate(cells):
        x, y, w, h = cell["x"], cell["y"], cell["w"], cell["h"]
        rx, ry = w / 2, h / 2
        area = w * h

        x_start = int(max(0, (x - rx) // bin_width))
        x_end = int(min(grid_size[0], (x + rx) // bin_width + 1))
        y_start = int(max(0, (y - ry) // bin_width))
        y_end = int(min(grid_size[1], (y + ry) // bin_width + 1))

        for i in range(x_start, x_end):
            for j in range(y_start, y_end):
                bin_x = i * bin_width + 0.5
                bin_y = j * bin_width + 0.5

                dx = (bin_x - x) / rx
                dy = (bin_y - y) / ry

                dphi_dx, dphi_dy = bump_grad_2d(dx, dy)
                overflow = max(0.0, density_grid[i, j] - target_density)

                grad_x = -area * density_weight * dphi_dx * overflow / rx
                grad_y = -area * density_weight * dphi_dy * overflow / ry
                gradients[idx][0] += grad_x
                gradients[idx][1] += grad_y

    return density_grid, gradients


# Animation setup
fig, ax = plt.subplots()
ims = []

cells = [dict(cell) for cell in initial_cells]

for step in range(steps):
    density, grads = compute_density_and_gradients(cells)

    # Apply gradient descent
    for i in range(len(cells)):
        cells[i]["x"] -= lr * grads[i][0]
        cells[i]["y"] -= lr * grads[i][1]

    im = ax.imshow(density.T, origin="lower", cmap="hot", animated=True)
    rects = []
    for cell in cells:
        rect = patches.Rectangle(
            (cell["x"] - cell["w"] / 2, cell["y"] - cell["h"] / 2),
            cell["w"],
            cell["h"],
            linewidth=1.5,
            edgecolor="blue",
            facecolor="none",
            animated=True,
        )
        ax.add_patch(rect)
        rects.append(rect)

    ims.append([im] + rects)

ani = animation.ArtistAnimation(fig, ims, interval=300, blit=True)
plt.close()

ani_path = "density_optimization.gif"
ani.save(ani_path, writer="pillow")
Image(ani_path)  # Display the saved animation
