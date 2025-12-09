import matplotlib.pyplot as plt
import numpy as np

# 設定整體字型與樣式（論文風格）
plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["axes.linewidth"] = 1.2  # 坐標軸線條粗細
plt.rcParams["pdf.fonttype"] = 42  # 確保字型在 PDF 中可編輯

# Data
x_labels = ["Testcase1", "Testcase2", "Testcase3", "Hidden1", "Hidden2", "Hidden3", "Hidden4"]
multi_core = [2.68, 5.43, 2.73, 4.63, 7.57, 4.19, 2.71]
single_core = [
    2.435,
    8.67 / 4.544,
    5.94 / 2.398,
    7.85 / 4.521,
    10.76 / 6.436,
    7.42 / 3.284,
    5.91 / 2.427,
]

single_core = [
    t * ratio for (t, ratio) in zip(multi_core, single_core)
]  # Adjust sequential runtimes

# X positions
x = np.arange(len(x_labels))

# Plot
plt.figure(figsize=(6.2, 4.2))
plt.plot(x, single_core, "s--", color="steelblue", label="Sequential", markersize=6, linewidth=1.8)
plt.plot(x, multi_core, "o--", color="darkorange", label="Parallel", markersize=6, linewidth=1.8)

# Labels and style
plt.xticks(x, x_labels, fontsize=11, rotation=20)
plt.yticks(fontsize=11)
plt.ylabel("Runtime (s)", fontsize=12)

# Legend and grid
plt.legend(loc="upper left", frameon=True, fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)

# Layout and export
plt.tight_layout()
plt.savefig("runtime_comparison_sequential_vs_parallel.pdf", dpi=72, bbox_inches="tight")
