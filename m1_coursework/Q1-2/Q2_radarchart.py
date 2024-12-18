import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from math import pi

# Load the CSV data
csv_file = "optuna_study_results.csv"
data = pd.read_csv(csv_file)

parameters = ["params_batch_size", "params_hidden1", "params_hidden2", "params_l1_lambda", "params_lr"]

# Normalize parameters
normalized_data = data.copy()
for param in parameters:
    normalized_data[param] = (data[param] - data[param].min()) / (data[param].max() - data[param].min())

def plot_radar(data, parameters, colors):
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'polar': True})

    num_vars = len(parameters)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]  # Close the circle

    # Plot each iteration
    for i, row in data.iterrows():
        values = [row[param] for param in parameters]
        values += values[:1]  # Close the circle
        ax.plot(angles, values, linewidth=3, linestyle='solid', label=f"Iteration {row['number']}", color=colors[i])

    # Add labels
    paramlabels = [p.replace("params_", "") for p in parameters]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(paramlabels)
    ax.set_yticks([])
    ax.set_title("Radar Chart of Parameters Over Iterations", fontsize=16)

    # plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.savefig("radarchart.png")
    plt.close(fig)

# Time-Series Plot
def plot_time_series(data, colors):
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, row in data.iterrows():
        ax.scatter(row["number"], row["value"], color=colors[i], label=f"Iteration {row['number']}", s=100)
    ax.plot(data["number"], data["value"], color="gray", alpha=0.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Model Performance (Value)")
    ax.set_title("Model Performance Over Iterations", fontsize=16)
    # ax.legend(loc="best")
    plt.savefig("timeseries.png")
    plt.close(fig)

# Generate colors for iterations
colors = plt.cm.PuRd(np.linspace(0, 1, len(data)))

# Plot radar chart
plot_radar(normalized_data, parameters, colors)

# Plot time-series
plot_time_series(data, colors)
