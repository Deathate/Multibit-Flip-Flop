import torch
import torch.nn as nn
import torch.optim as optim

# Smoothing factor gamma
gamma = 1.0

# Initialize x positions of 3 pins (requires gradient)
x = torch.tensor([5.0, 10.0], requires_grad=True)

# Optimizer
optimizer = optim.Adam([x], lr=0.1)


# Weighted Average Wirelength Function
def weighted_average_wirelength(x, gamma):
    exp_pos = torch.exp(x / gamma)
    exp_neg = torch.exp(-x / gamma)
    pos_term = torch.sum(x * exp_pos) / torch.sum(exp_pos)
    neg_term = torch.sum(x * exp_neg) / torch.sum(exp_neg)
    return pos_term - neg_term


# Optimization loop
for step in range(100):
    optimizer.zero_grad()
    wl = weighted_average_wirelength(x, gamma)
    wl.backward()
    optimizer.step()

    if step % 10 == 0:
        print(f"Step {step:3d}: Wirelength = {wl.item():.4f}, x = {x.data.tolist()}")

print("\nFinal optimized x-positions:", x.data.tolist())
