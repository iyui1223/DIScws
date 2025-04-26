import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
from preprocessor import read_hdf5
from scipy.stats import pearsonr

def evaluate_RMSE_ACC(ground_truth, predictions, exp_name):
    """
    Evaluate RMSE, Bias, and ACC for each time step separately
    while handling missing predictions (NaN values).
    """
    num_systems, tsteps, types = predictions.shape
    in_steps = tsteps // 2

    # Initialize arrays for time-series evaluation
    rmse_t = np.full((tsteps - in_steps, types), np.nan)
    bias_t = np.full((tsteps - in_steps, types), np.nan)
    acc_t = np.full((tsteps - in_steps, types), np.nan)

    for t in range(in_steps, tsteps):  # Only evaluate after prediction starts
        for j in range(types):  # 0 = Prey, 1 = Predator
            valid_indices = ~np.isnan(predictions[:, t, j]) & ~np.isnan(ground_truth[:, t, j])

            if np.any(valid_indices):
                rmse_t[t - in_steps, j] = np.sqrt(np.mean((predictions[valid_indices, t, j] -
                                                            ground_truth[valid_indices, t, j]) ** 2))
                bias_t[t - in_steps, j] = np.mean(predictions[valid_indices, t, j] -
                                                  ground_truth[valid_indices, t, j])

                ground_anomaly = ground_truth[:, t, j] - np.nanmean(ground_truth[:, t, j])
                pred_anomaly = predictions[:, t, j] - np.nanmean(predictions[:, t, j])

                valid_anomaly = ~np.isnan(ground_anomaly) & ~np.isnan(pred_anomaly)
                if np.any(valid_anomaly):
                    acc_t[t - in_steps, j] = pearsonr(ground_anomaly[valid_anomaly],
                                                      pred_anomaly[valid_anomaly])[0]

    # Plot time-series RMSE, Bias, and ACC
    fig, axs = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    labels = ["Prey", "Predator"]
    time_range = np.arange(in_steps, tsteps)

    for i, (metric, title) in enumerate(zip([rmse_t, bias_t, acc_t], ["RMSE", "Bias", "ACC"])):
        ax = axs[i]
        if title in ["RMSE", "Bias"]:
            ax2 = ax.twinx()
            ax.plot(time_range, metric[:, 0], label=f"Prey", color="tab:blue", marker=".")
            ax.set_ylabel(f"{title} (Prey)", color="tab:blue")
            ax.tick_params(axis='y', labelcolor="tab:blue")

            ax2.plot(time_range, metric[:, 1], label=f"Predator", color="tab:red", linestyle="dashed", marker="x")
            ax2.set_ylabel(f"{title} (Predator)", color="tab:red")
            ax2.tick_params(axis='y', labelcolor="tab:red")

            lines, labels_ = ax.get_legend_handles_labels()
            lines2, labels2_ = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels_ + labels2_, loc="upper right")

        else:  # ACC: single y-axis
            ax.plot(time_range, metric[:, 0], label="Prey", color="tab:blue", marker=".")
            ax.plot(time_range, metric[:, 1], label="Predator", color="tab:red", linestyle="dashed", marker="x")
            ax.set_ylabel("ACC")
            ax.legend()
        ax.set_title(title)
        ax.grid(True)

    axs[-1].set_xlabel("Time Steps")
    plt.tight_layout()
    plt.savefig(f"{exp_name}_metrics.png")

    # Compute and compare lag correlations
    max_lag = 10  # Maximum lag to consider
    lags = np.arange(-max_lag, max_lag + 1)
    lag_corr_ground = np.full((num_systems, len(lags)), np.nan)
    lag_corr_pred = np.full((num_systems, len(lags)), np.nan)

    for i in range(num_systems):
        prey_truth = ground_truth[i, in_steps:, 0]
        predator_truth = ground_truth[i, in_steps:, 1]
        prey_pred = predictions[i, in_steps:, 0]
        predator_pred = predictions[i, in_steps:, 1]

        for j, lag in enumerate(lags):
            if lag < 0:
                valid_indices = ~np.isnan(prey_truth[:lag]) & ~np.isnan(predator_truth[-lag:])
                if np.any(valid_indices):
                    lag_corr_ground[i, j] = pearsonr(prey_truth[:lag][valid_indices],
                                                     predator_truth[-lag:][valid_indices])[0]

                valid_indices = ~np.isnan(prey_pred[:lag]) & ~np.isnan(predator_pred[-lag:])
                if np.any(valid_indices):
                    lag_corr_pred[i, j] = pearsonr(prey_pred[:lag][valid_indices],
                                                   predator_pred[-lag:][valid_indices])[0]

            elif lag > 0:
                valid_indices = ~np.isnan(prey_truth[lag:]) & ~np.isnan(predator_truth[:-lag])
                if np.any(valid_indices):
                    lag_corr_ground[i, j] = pearsonr(prey_truth[lag:][valid_indices],
                                                     predator_truth[:-lag][valid_indices])[0]

                valid_indices = ~np.isnan(prey_pred[lag:]) & ~np.isnan(predator_pred[:-lag])
                if np.any(valid_indices):
                    lag_corr_pred[i, j] = pearsonr(prey_pred[lag:][valid_indices],
                                                   predator_pred[:-lag][valid_indices])[0]

            else:
                valid_indices = ~np.isnan(prey_truth) & ~np.isnan(predator_truth)
                if np.any(valid_indices):
                    lag_corr_ground[i, j] = pearsonr(prey_truth[valid_indices],
                                                     predator_truth[valid_indices])[0]

                valid_indices = ~np.isnan(prey_pred) & ~np.isnan(predator_pred)
                if np.any(valid_indices):
                    lag_corr_pred[i, j] = pearsonr(prey_pred[valid_indices],
                                                   predator_pred[valid_indices])[0]

    # Compute average across systems, ignoring NaNs
    avg_lag_corr_ground = np.nanmean(lag_corr_ground, axis=0)
    avg_lag_corr_pred = np.nanmean(lag_corr_pred, axis=0)

    # Plot lag correlations
    plt.figure(figsize=(6, 4))
    plt.plot(lags, avg_lag_corr_ground, label="Ground Truth", marker="o")
    plt.plot(lags, avg_lag_corr_pred, label="Prediction", marker="s", linestyle="dashed")
    plt.axhline(0, color='black', linestyle='dotted')
    plt.axvline(0, color='black', linestyle='dotted')
    plt.xlabel("Lag (Time Steps)")
    plt.ylabel("Correlation Coefficient")
    plt.title("Lag Correlation between Prey and Predator")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{exp_name}_LagCorr.png")

    return

def sample_plots(ground_truth, predictions, exp_name):
    """
    Compare and visualize input vs. prediction
    """
    num_systems, tsteps, types = predictions.shape
    in_steps = tsteps // 2
    labels = ["Prey", "Predator"]

    valid_indices = []
    for i in range(num_systems):
        if not np.all(np.isnan(predictions[i, in_steps:, :])):
            valid_indices.append(i)
        if len(valid_indices) >= 10:
            break
    print(valid_indices)


    for i in valid_indices:
        fig, axes = plt.subplots(2, 1, figsize=(5, 3), sharex=True)
        for j in range(2):
            axes[j].plot(range(tsteps), ground_truth[i, :, j], label=f"Ground Truth {labels[j]}", color='blue')
            pred_tsteps = np.sum(~np.isnan(predictions[i, in_steps:, j]))
            axes[j].plot(
                range(in_steps, in_steps + pred_tsteps),
                predictions[i, in_steps:in_steps + pred_tsteps, j],
                label=f"Predicted {labels[j]}",
                color='red',
                linestyle='dashed'
            )
            axes[j].axvline(x=in_steps, color='black', linestyle='dotted')
            axes[j].set_ylabel(labels[j])
            axes[j].legend()

        axes[1].set_xlabel("Time Steps")
        plt.suptitle(f"System {i}: Input vs. Prediction")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"{exp_name}_plot{i}.png")
        plt.close(fig)


if __name__ == "__main__":
    # exp_name -- should be passed through user interface from wrapper shell
    parser = argparse.ArgumentParser(description="Run predictions with Qwen model")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
    args = parser.parse_args()
    prediction_dir = "./predictions/"+args.exp_name

    time, ground_truth = read_hdf5("lotka_volterra_data.h5")
    num_systems ,tsteps, types = ground_truth.shape
    
    # Process each system independently
    num_pred = 1000
    in_steps = tsteps // 2 
    predicted_steps = tsteps - in_steps

    predictions = np.zeros((num_systems, tsteps, types))
    predictions *= np.nan

    for i in range(num_pred):
        pred_file = os.path.join(prediction_dir, f"system{i}.npy")
        if os.path.exists(pred_file):
            pred = np.load(pred_file)
            if pred.shape[1] >= predicted_steps - 9:
                predictions[i,predicted_steps:-9,:] = pred[0, :predicted_steps-9,:] 
            print(f"prediction file {pred_file} used {pred[0, predicted_steps-19:predicted_steps-9, 0]}")
    exp_name = args.exp_name
    evaluate_RMSE_ACC(ground_truth, predictions, exp_name.replace(".", ""))
    sample_plots(ground_truth, predictions, exp_name.replace(".", ""))

