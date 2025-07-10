# %%
import matplotlib.pyplot as plt

# Data
categories = ["10%", "20%", "30%", "100%"]
values = [9923.66, 8169.07, 7936.47, 7557]
times = [94, 137, 172, 447]

# Plot
plt.figure(figsize=(8, 5))
plt.plot(times, values, marker="o")
for i, cat in enumerate(categories):
    plt.text(times[i], values[i], cat, fontsize=9, ha="right")

plt.xlabel("Time (s)")
plt.ylabel("Timing ")
plt.title("Timing vs. Time")
plt.grid(True)
plt.tight_layout()
plt.show()
