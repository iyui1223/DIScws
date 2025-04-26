import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Load the dataset
file_path = "temp.png"

# Simulated DataFrame based on user's text description
df = pd.read_csv("log_rx_lr1-4e_s1000_cl512.txt", sep="\t")

smoothed = df.rolling(window=50, min_periods=1).mean()

# Plot every other row (validation rows) after smoothing
smoothed_downsampled = smoothed.iloc[::2]

# Plotting
num_steps = smoothed_downsampled.shape[0]
x_ticks = np.linspace(0, num_steps - 1, num_steps)

# Plotting
plt.figure(figsize=(5, 3))
for col in smoothed_downsampled.columns:
    plt.plot(x_ticks, smoothed_downsampled[col], label=col)

plt.xlabel("Steps")
plt.ylabel("Smoothed Loss (log scale)")
plt.title("Smoothed Loss Curve over Training Steps")
plt.yscale("log")  # Logarithmic scale for y-axis
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(file_path)
