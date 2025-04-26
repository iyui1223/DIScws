#!/bin/bash
#SBATCH --job-name=diffusion_opt
#SBATCH --output=../Log/output.log
#SBATCH --error=../Log/error.log
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=cclake
#SBATCH --cpus-per-task=16
#SBATCH -A MPHIL-DIS-SL2-CPU

set -eox

Root="/path/to/your/root/dir"
cd ${Root}

exp_name="parallel-omp"
opt_levels=("-O0" "-O3")

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-icl

export I_MPI_PIN_DOMAIN=omp:compact
export I_MPI_PIN_ORDER=scatter

rm -rf build
mkdir -p build

for numthreads in 1 2 4 8 16; do
    export OMP_NUM_THREADS=${numthreads}
    echo "Starts loop for OpenMP threads: ${numthreads}"

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

        for matrix_size in 100 1000; do
            echo "Running ${exp_name} with matrix size ${matrix_size}, threads=${numthreads}"
            ./${exp_name} ${matrix_size} ${matrix_size} "100"
        done

        mkdir -p "${Root}/Data/${exp_name}_${opt_options}_nt${numthreads}"
        cp output/* "${Root}/Data/${exp_name}_${opt_options}_nt${numthreads}"

        cd "${Root}"
    done
done
