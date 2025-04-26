#!/bin/bash
#SBATCH --job-name=diffusion_opt
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --time=00:05:00                # Max execution time (HH:MM:SS)
#SBATCH --nodes=1
#SBATCH --partition=cclake
#SBATCH --mem=8G                        # Memory request
#SBATCH --cpus-per-task=1               # Request CPU cores
#SBATCH -A MPHIL-DIS-SL2-CPU            # Your project account

set -eox  # Exit on error, print commands

# Define root directory
Root="/path/to/your/root/dir"
cd ${Root}

# Define experiments and optimization levels
exp_names=("original" "non-parallel")
opt_levels=("-O0" "-O1" "-O2" "-O3" "-Ofast")

# Clear build directory ONCE at the start
rm -rf build
mkdir -p build

# Loop over optimization levels
for opt_options in "${opt_levels[@]}"; do
    for exp_name in "${exp_names[@]}"; do
        export EXP_NAME=${exp_name} 
        
        echo "---------------------------"
        echo "Building: ${exp_name} with ${opt_options}"
        echo "---------------------------"

        # Configure CMake
        # cmake -DOPTIMIZATION_LEVEL="${opt_options}" -B build -DBUILD_${exp_name}=ON -DPROFILE=ON
        cmake -B build -DVALGRIND=ON -DBUILD_${exp_name}=ON
        cmake --build build

        # Prepare work directory
        mkdir -p "${Root}/Work"
        cd "${Root}/Work"
        ln -sf "${Root}/Data/experiment_log.csv" .
        ln -sf "${Root}/Bin/${exp_name}" .

        # Run experiment for different grid sizes
        for matrix_size in "100" "1000"; do
            echo "Running ${exp_name} with matrix size ${matrix_size}"
            ./${exp_name} ${matrix_size} ${matrix_size} "100"
        done

        # Save output results
        mkdir -p "${Root}/Data/${exp_name}_${opt_options}"
        cp output/* "${Root}/Data/${exp_name}_${opt_options}"

        # Move back to root directory
        cd "${Root}"
    done
done
