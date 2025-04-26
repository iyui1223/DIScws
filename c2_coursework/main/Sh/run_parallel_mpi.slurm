#!/bin/bash
#SBATCH --job-name=diffusion_opt
#SBATCH --output=../Log/output.log
#SBATCH --error=../Log/error.log
#SBATCH --time=00:30:00                # Max execution time (HH:MM:SS)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=cclake
#SBATCH --cpus-per-task=16               # Request CPU cores
#SBATCH -A MPHIL-DIS-SL2-CPU            # Your project account

set -eox  # Exit on error, print commands

# Define root directory
Root="/path/to/your/root/dir"
cd ${Root}

# Define experiments and optimization levels
exp_name="parallel-mpi"
opt_levels=("-O0" "-O3")

numnodes=$SLURM_JOB_NUM_NODES
max_numtasks=$SLURM_NTASKS

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-icl

export I_MPI_PIN_DOMAIN=omp:compact # Domains are $OMP_NUM_THREADS cores in size
export I_MPI_PIN_ORDER=scatter # Adjacent domains have minimal sharing of caches/sockets
export OMP_NUM_THREADS=1

# Clear build directory ONCE at the start
rm -rf build
mkdir -p build

# Loop over optimization levels + number of processes
for numprocess in 1 2 4 8 16; do
    echo "starts loop for mpi num ${numprocess}" 
    for opt_options in "${opt_levels[@]}"; do
        export EXP_NAME=${exp_name} 
        
        echo "---------------------------"
        echo "Building: ${exp_name} with ${opt_options}"
        echo "---------------------------"

        cmake -DOPTIMIZATION_LEVEL="${opt_options}" -B build -DBUILD_${exp_name}=ON
        cmake --build build

        mkdir -p "${Root}/Work"
        cd "${Root}/Work"
        ln -sf "${Root}/Data/experiment_log.csv" .
        ln -sf "${Root}/Bin/${exp_name}" .

        # Run experiment for different grid sizes
        for matrix_size in 100 1000 10000; do
            echo "Running ${exp_name} with matrix size ${matrix_size}, np=${numprocess}"
            if [[ "$exp_name" == parallel-mpi* ]]; then
                mpirun -np ${numprocess} ./${exp_name} ${matrix_size} ${matrix_size} "100"
            else
                ./${exp_name} ${matrix_size} ${matrix_size} "100"
            fi
        done

        mkdir -p "${Root}/Data/${exp_name}_${opt_options}_np${numprocess}"
        cp output/* "${Root}/Data/${exp_name}_${opt_options}_np${numprocess}"

        cd "${Root}"
    done
done
