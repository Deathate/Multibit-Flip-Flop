import os
import sys
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

# Device configuration (CPU for now)
device = torch.device("cpu")


# Quartic bump function and gradient (vectorized)
def bump_1d(u):
    mask = (u.abs() <= 1).float()
    return ((1 - u**2) ** 2) * mask


# Grid and parameters
grid_size = (100, 100)
bin_width = 1
target_density = 0.2
density_weight = 1.0
lr = 0.1
steps = 200

# Define initial cell tensors: [x, y, w, h]
# cells = torch.tensor(
#     [[20.0, 25.0, 8.0, 4.0], [23.0, 25.0, 4.0, 4.0], [22.0, 24.0, 4.0, 4.0]],
#     requires_grad=True,
#     device=device,
# )
# create random cells
num_cells = 100
num_fixed_cells = 30
dynamic_cells = []
for i in range(num_cells):
    x = torch.randint(int(grid_size[0] * 0.3), int(grid_size[0] * 0.7), (1,), device=device).item()
    y = torch.randint(int(grid_size[1] * 0.3), int(grid_size[1] * 0.7), (1,), device=device).item()
    w = torch.randint(3, 10, (1,), device=device).item()
    h = torch.randint(5, 10, (1,), device=device).item()
    dynamic_cells.append([x, y, w, h])
fixed_cells = []
for i in range(num_fixed_cells):
    x = torch.randint(int(grid_size[0] * 0.1), int(grid_size[0] * 0.9), (1,), device=device).item()
    y = torch.randint(int(grid_size[1] * 0.1), int(grid_size[1] * 0.9), (1,), device=device).item()
    w = torch.randint(15, 20, (1,), device=device).item()
    h = torch.randint(15, 20, (1,), device=device).item()
    fixed_cells.append([x, y, w, h])
fixed_cells = torch.tensor(fixed_cells, requires_grad=False, device=device, dtype=torch.float32)
cells = torch.tensor(dynamic_cells, requires_grad=True, device=device, dtype=torch.float32)
# Bin centers
x_coords = torch.arange(grid_size[0], dtype=torch.float32, device=device) * bin_width + 0.5
y_coords = torch.arange(grid_size[1], dtype=torch.float32, device=device) * bin_width + 0.5
bin_x, bin_y = torch.meshgrid(x_coords, y_coords, indexing="ij")
# Animation setup
fig, ax = plt.subplots()
ims = []


# Optimization loop
def aggregate_density_values(density, cells, bin_x, bin_y):
    for i in range(cells.shape[0]):
        x, y, w, h = cells[i]
        rx, ry = w / 2, h / 2
        area = w * h

        dx = (bin_x - x) / rx
        dy = (bin_y - y) / ry

        phi = bump_1d(dx) * bump_1d(dy)
        phi_sum = phi.sum() * bin_width**2
        density += area * phi / phi_sum if phi_sum > 0 else torch.zeros_like(density)


for step in range(steps):
    density = torch.zeros(grid_size, dtype=torch.float32, device=device)
    aggregate_density_values(density, cells, bin_x, bin_y)
    aggregate_density_values(density, fixed_cells, bin_x, bin_y)
    # Compute penalty and backward
    overflow = torch.clamp(density - target_density, min=0.0)
    mask = (overflow > 0).float()
    overflow = overflow * mask
    penalty = (overflow**2).sum() * density_weight
    print(f"Step {step + 1}, Penalty: {penalty}")
    penalty.backward()
    # print(cells.grad[:, 0:2])

    # Update cells
    if cells.grad is not None:
        with torch.no_grad():
            cells[:, 0] -= (lr * cells.grad[:, 0]).clamp(max=10, min=-10)  # Prevent too large updates
            cells[:, 1] -= (lr * cells.grad[:, 1]).clamp(max=10, min=-10)
            cells.grad.zero_()
            # add random noise to prevent local minima
            cells[:, 0:2] += torch.randn_like(cells[:, 0:2]) * 0.1

    # Convert to NumPy for visualization
    dens_np = density.detach().cpu().numpy().T
    im = ax.imshow(dens_np, origin="lower", cmap="hot", animated=True)
    rects = []
    for i in range(cells.shape[0]):
        x, y, w, h = cells[i].detach().cpu().numpy()
        rect = patches.Rectangle(
            (x - w / 2, y - h / 2),
            w,
            h,
            linewidth=1.5,
            edgecolor="blue",
            facecolor="none",
            animated=True,
        )
        ax.add_patch(rect)
        rects.append(rect)

    ims.append([im] + rects)

# Save animation
ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True)
plt.close()

ani_path = Path(__file__).parent / "density_optimization.gif"
ani.save(ani_path, writer="pillow")
print(f"Animation saved to {ani_path}")
