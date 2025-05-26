import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from utility_image_wo_torch import *

gamma = 1.0

# All pins (5), each has (x, y)
init_all = torch.tensor(
    [
        [15.0, 2.0],  # pin0 — fixed
        [5.0, 2.0],  # pin1 — movable
        [10.0, 3.0],  # pin2 — movable
        [15.0, 1.0],  # pin3 — movable
        [20.0, 0.0],  # pin4 — fixed
    ],
    dtype=torch.float32,
)
init_all_np = init_all.detach().clone().numpy()

# Define which pins to fix
fixed_indices = [0, 4]
movable_indices = [1, 2, 3]

# Separate movable and fixed parts
movable_pos = init_all[movable_indices].clone().detach().requires_grad_(True)
fixed_pos = init_all[fixed_indices]

nets = [[0, 1], [1, 2, 3], [3, 4]]


# WAWL 1D function
def wawl_1d(pin_positions, gamma):
    exp_pos = torch.exp(pin_positions / gamma)
    exp_neg = torch.exp(-pin_positions / gamma)
    pos_term = torch.sum(pin_positions * exp_pos) / torch.sum(exp_pos)
    neg_term = torch.sum(pin_positions * exp_neg) / torch.sum(exp_neg)
    return pos_term - neg_term


# Total wirelength with full positions (movable + fixed)
def total_wirelength_from_movable(movable, fixed, movable_idx, fixed_idx, nets, gamma):
    N = len(movable) + len(fixed)
    full_pos = torch.zeros((N, 2), dtype=movable.dtype)
    full_pos[movable_idx] = movable
    full_pos[fixed_idx] = fixed

    total = 0.0
    for net in nets:
        coords = full_pos[net]
        total += wawl_1d(coords[:, 0], gamma) + wawl_1d(coords[:, 1], gamma)
    return total, full_pos


# Optimizer
optimizer = optim.Adam([movable_pos], lr=0.1)

history = []
for step in range(100):
    optimizer.zero_grad()
    wl, full_pos = total_wirelength_from_movable(
        movable_pos, fixed_pos, movable_indices, fixed_indices, nets, gamma
    )
    wl.backward()
    optimizer.step()

    history.append(wl.item())
    if step % 10 == 0:
        print(f"Step {step:3d}: Wirelength = {wl.item():.4f}")

final_pos = full_pos.detach().numpy()
initial_pos = init_all.numpy()


def plot_2d_movement(initial, final, nets, title):
    plt.figure(figsize=(8, 6))
    # Plot initial positions
    plt.scatter(initial[:, 0], initial[:, 1], color="blue", label="Initial", s=100)
    for i, (x, y) in enumerate(initial):
        plt.text(x, y + 0.3, f"pin{i}", color="blue", ha="center")

    # Plot final positions
    plt.scatter(final[:, 0], final[:, 1], color="green", label="Final", s=100)
    for i, (x, y) in enumerate(final):
        plt.text(x, y - 0.3, f"pin{i}", color="green", ha="center")

    # Draw arrows from initial to final positions
    # for i in range(len(initial)):
    #     dx = final[i, 0] - initial[i, 0]
    #     dy = final[i, 1] - initial[i, 1]
    #     plt.arrow(
    #         initial[i, 0],
    #         initial[i, 1],
    #         dx,
    #         dy,
    #         head_width=0.01,
    #         head_length=0.01,
    #         fc="orange",
    #         ec="orange",
    #     )

    # Draw net connections (after optimization)
    for net in nets:
        coords = final[net]
        plt.plot(coords[:, 0], coords[:, 1], "gray", linestyle="--")

    plt.title(title)
    plt.grid(True)
    # plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plot_images(plt.gcf(), 700)


# Call the function
plot_2d_movement(init_all_np, final_pos, nets, "2D Pin Optimization with Movement Arrows")
