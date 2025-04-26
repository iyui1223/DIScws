#!/bin/bash
#SBATCH --job-name=qwen_lora_train
#SBATCH --output=/rds/user/yi260/hpc-work/cwk/cwssubmit/m2/yi260/main/log/output.log
#SBATCH --error=/rds/user/yi260/hpc-work/cwk/cwssubmit/m2/yi260/main/log/error.log
#SBATCH --time=00:20:00                # Max execution time (HH:MM:SS)
#SBATCH --partition=ampere             # GPU partition # pvc
#SBATCH --gres=gpu:4                    # Request 1 GPU
#SBATCH -A MPHIL-DIS-SL2-GPU            # Your project account # TRAINING-DAWN-GPU

# Wrapper script for execution predict.sh
##################
#### editable ####
##################
exp_name="lora-train"

set -eox
# source ~/.bashrc
source "/usr/local/software/archive/linux-scientific7-x86_64/gcc-9/miniconda3-4.7.12.1-rmuek6r3f6p3v6fdj7o2klyzta3qhslh/etc/profile.d/conda.sh"

# if it was DAWS
# module purge
# module load default-dawn

conda init bash
conda activate m2coursework

# Define directories
ROOT_DIR="/rds/user/yi260/hpc-work/cwk/cwssubmit/m2/yi260/main"
# ROOT_DIR=$(dirname "$(realpath "$0")")/..
WORKDIR="${ROOT_DIR}/workdir"
SRC_DIR="${ROOT_DIR}/src"
DATA_DIR="${ROOT_DIR}/data/"
PREDICTIONS_DIR="${DATA_DIR}/predictions/${exp_name}"
FIGS_DIR="${ROOT_DIR}/figs/predictions/${exp_name}"
LOG_DIR="${ROOT_DIR}/log"

mkdir -p "$WORKDIR" "$LOG_DIR" "$PREDICTIONS_DIR" "$FIGS_DIR"
cd "$WORKDIR"

echo "Setting up working directory..."

ln -sf "${SRC_DIR}/"*.py "$WORKDIR/"
ln -sf "${DATA_DIR}/lotka_volterra_data.h5" "$WORKDIR/"
ln -sf "${DATA_DIR}/predictions" "$WORKDIR/"

echo "Running training process..."

python3 lora_skeleton.py

echo "Organizing output files..."
mv -v system*.npy "$PREDICTIONS_DIR/"
mv -v *.png "$FIGS_DIR

echo "All files have been moved successfully."
