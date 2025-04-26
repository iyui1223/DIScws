def stats(infile):
    """
    Calculate statistics of the dataset and plot them. 
    """
    from preprocessor import read_hdf5
    import matplotlib.pyplot as plt
    import numpy as np
    time, trajectories = read_hdf5(infile)
    
    # Trajectories for 3 selected prey-predator series
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ii = 1
    for i in range(0, 1000, 333):
        color = colors[ii-0]  # Get the next color in the cycle
        plt.plot(trajectories[i, :100, 0], label=f"Prey {ii}", color=color)
        plt.plot(trajectories[i, :100, 1], label=f"Predator {ii}", linestyle="dashed", color=color)
        ii += 1

    plt.xlabel("Time Step")
    plt.ylabel("Population")
    plt.title("Predator-Prey Model Trajectories")
    plt.legend()
    plt.grid(True)
    plt.savefig("trajectories.png")

    # Compute standard deviations and means
    stn_dev = np.zeros((trajectories.shape[0], trajectories.shape[2]))
    _mean = np.zeros((trajectories.shape[0], trajectories.shape[2]))

    for i in range(trajectories.shape[0]):
        for j in range(2):
            stn_dev[i, j] = np.std(trajectories[i, :, j])
            _mean[i, j] = np.mean(trajectories[i, :, j])

    # Extracting prey and predator statistics
    prey_mean = _mean[:, 0]
    prey_std = stn_dev[:, 0]
    predator_mean = _mean[:, 1]
    predator_std = stn_dev[:, 1]

    # Create the scatter plot with marginal histograms
    fig, axs = plt.subplot_mosaic([['histx', '.'],
                                ['scatter', 'histy']],
                                figsize=(8, 8),
                                width_ratios=(4, 1), height_ratios=(1, 4),
                                layout='constrained')

    # 2D Histogram for Prey
    hb_prey = axs['scatter'].hist2d(prey_mean, prey_std, bins=30, cmap='Blues')
    fig.colorbar(hb_prey[3], ax=axs['scatter'], label="Count")

    # Histograms for marginal distributions (Prey)
    axs['histx'].hist(prey_mean, bins=30, color='blue', alpha=0.7)
    axs['histy'].hist(prey_std, bins=30, color='blue', alpha=0.7, orientation='horizontal')

    # Labels
    axs['scatter'].set_xlabel("Mean (Prey)")
    axs['scatter'].set_ylabel("Std Dev (Prey)")
    axs['histx'].set_ylabel("Count")
    axs['histy'].set_xlabel("Count")

    plt.savefig("histo_prey.png")

    # Create another scatter plot for Predator statistics
    fig, axs = plt.subplot_mosaic([['histx', '.'],
                                ['scatter', 'histy']],
                                figsize=(8, 8),
                                width_ratios=(4, 1), height_ratios=(1, 4),
                                layout='constrained')

    # 2D Histogram for Predator
    hb_predator = axs['scatter'].hist2d(predator_mean, predator_std, bins=30, cmap='Reds')
    fig.colorbar(hb_predator[3], ax=axs['scatter'], label="Count")

    # Histograms for marginal distributions (Predator)
    axs['histx'].hist(predator_mean, bins=30, color='red', alpha=0.7)
    axs['histy'].hist(predator_std, bins=30, color='red', alpha=0.7, orientation='horizontal')

    # Labels
    axs['scatter'].set_xlabel("Mean (Predator)")
    axs['scatter'].set_ylabel("Std Dev (Predator)")
    axs['histx'].set_ylabel("Count")
    axs['histy'].set_xlabel("Count")

    plt.savefig("histo_predator.png")

    # Histogram
    # Flatten the prey and predator data
    prey = trajectories[:, :, 0].flatten()
    predator = trajectories[:, :, 1].flatten()

    percentile_q = 99
    prey_q = np.percentile(prey, percentile_q)
    predator_q = np.percentile(predator, percentile_q)

    # Create the histogram plot
    plt.figure(figsize=(8, 6))
    plt.hist(prey, bins=50, alpha=0.6, color='blue', label='Prey')
    plt.hist(predator, bins=50, histtype='step', color='red', linewidth=2, label='Predator')

    plt.axvline(prey_q, color='blue', linestyle='dashed', linewidth=2, label=f'Prey {percentile_q}th %ile = {prey_q:9.2f}')
    plt.axvline(predator_q, color='red', linestyle='dashed', linewidth=2, label=f'Predator {percentile_q}th %ile = {predator_q:9.2f}')

    plt.xlabel("Population Values")
    plt.ylabel("Frequency")
    plt.title("Histogram of Prey and Predator Populations with 99th Percentile Markers")
    plt.legend()
    plt.savefig("histo.png")

    
if __name__ == "__main__":
    infile="../data/lotka_volterra_data.h5"
    stats(infile)

    