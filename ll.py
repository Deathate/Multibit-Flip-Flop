import matplotlib.pyplot as plt
import mpld3

# Create a Matplotlib plot
fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4], [10, 20, 25, 30])

# Show the plot interactively
mpld3.show()
