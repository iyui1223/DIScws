#!/bin/bash
#SBATCH --job-name=evaluate_model
#SBATCH --output=/rds/user/yi260/hpc-work/cwk/cwssubmit/m2/yi260/main/Log/evaluate_output.log
#SBATCH --error=/rds/user/yi260/hpc-work/cwk/cwssubmit/m2/yi260/main/Log/evaluate_error.log
#SBATCH --time=01:00:00                # Max execution time (HH:MM:SS)
#SBATCH --nodes=1
#SBATCH --partition=cclake
#SBATCH --cpus-per-task=4               # Request CPU cores
#SBATCH -A MPHIL-DIS-SL2-CPU            # Your project account

# exp_name="original"
exp_name="lora_r8_lr0.0001_s1000_cl512"

set -eox

source "/usr/local/software/archive/linux-scientific7-x86_64/gcc-9/miniconda3-4.7.12.1-rmuek6r3f6p3v6fdj7o2klyzta3qhslh/etc/profile.d/conda.sh"
conda init bash
conda activate m2coursework

# Define directories
ROOT_DIR="/rds/user/yi260/hpc-work/cwk/cwssubmit/m2/yi260/main"
WORKDIR="${ROOT_DIR}/workdir"
SRC_DIR="${ROOT_DIR}/src"
DATA_DIR="${ROOT_DIR}/data/"
PREDICTIONS_DIR="${DATA_DIR}/predictions/${exp_name}"
FIGS_DIR="${ROOT_DIR}/figs/evaluations/${exp_name}"
LOG_DIR="${ROOT_DIR}/log"

mkdir -p "$WORKDIR" "$LOG_DIR" "$FIGS_DIR"
cd "$WORKDIR"

echo "Setting up working directory..."

ln -sf "${SRC_DIR}/"*.py "$WORKDIR/"
ln -sf "${DATA_DIR}/lotka_volterra_data.h5" "$WORKDIR/"
ln -sf "${DATA_DIR}/predictions" "$WORKDIR/"

echo "Running evaluation process..."

python3 evaluate.py --exp_name "${exp_name}"

echo "Evaluation completed."

echo "Organizing output files..."
mv -v *.png "$FIGS_DIR"

echo "All files have been moved successfully."

