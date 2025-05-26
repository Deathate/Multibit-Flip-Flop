import matplotlib.pyplot as plt

# Initial and final positions
initial_positions = [0.0, 5.0, 10.0, 15.0, 20.0]
final_positions = x.detach().numpy()

# Nets
nets = [[0, 1], [1, 2, 3], [3, 4]]  # Net 0  # Net 1  # Net 2


def plot_positions(positions, title):
    plt.figure(figsize=(10, 2))
    y = [0] * len(positions)  # All on the same horizontal line

    # Plot pin positions
    plt.scatter(positions, y, color="blue", s=100)
    for i, xpos in enumerate(positions):
        plt.text(xpos, 0.1, f"pin{i}", ha="center", fontsize=9)

    # Draw nets as horizontal lines connecting pins
    for net in nets:
        net_x = [positions[i] for i in net]
        net_x.sort()
        plt.plot(net_x, [0] * len(net_x), color="gray", linewidth=1)

    plt.title(title)
    plt.yticks([])
    plt.xlabel("x-coordinate")
    plt.grid(True, axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


# Plot before and after optimization
plot_positions(initial_positions, "🔵 Initial Pin Positions")
plot_positions(final_positions, "🟢 Optimized Pin Positions")
