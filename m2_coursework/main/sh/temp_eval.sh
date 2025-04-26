#!/bin/bash
#SBATCH --job-name=eval_r2_cl512
#SBATCH --output=/rds/user/yi260/hpc-work/cwk/cwssubmit/m2/yi260/main/log/eval_r2_cl512_output.log
#SBATCH --error=/rds/user/yi260/hpc-work/cwk/cwssubmit/m2/yi260/main/log/eval_r2_cl512_error.log
#SBATCH --time=00:08:00
#SBATCH --nodes=1
#SBATCH --partition=cclake
#SBATCH --cpus-per-task=4
#SBATCH -A MPHIL-DIS-SL2-CPU

set -eox

source "/usr/local/software/archive/linux-scientific7-x86_64/gcc-9/miniconda3-4.7.12.1-rmuek6r3f6p3v6fdj7o2klyzta3qhslh/etc/profile.d/conda.sh"
conda init bash
conda activate m2coursework

ROOT_DIR="/rds/user/yi260/hpc-work/cwk/cwssubmit/m2/yi260/main"
WORKDIR="${ROOT_DIR}/workdir"
SRC_DIR="${ROOT_DIR}/src"
DATA_DIR="${ROOT_DIR}/data/"
PREDICTIONS_DIR="${DATA_DIR}/predictions/lora_r2_lr1e-05_s5000_cl512"
FIGS_DIR="${ROOT_DIR}/figs/evaluations/"
LOG_DIR="${ROOT_DIR}/log"

mkdir -p "$WORKDIR" "$LOG_DIR" "$FIGS_DIR"
cd "$WORKDIR"

echo "Setting up working directory..."

ln -sf "${SRC_DIR}/"*.py "$WORKDIR/"
ln -sf "${DATA_DIR}/lotka_volterra_data.h5" "$WORKDIR/"
ln -sf "${DATA_DIR}/predictions" "$WORKDIR/"

echo "Running evaluation process..."

python3 evaluate.py --exp_name "lora_r2_lr1e-05_s5000_cl512"

echo "Evaluation completed."

echo "Organizing output files..."
mkdir -p "$FIGS_DIR/temp"
mv -v *.png "$FIGS_DIR/temp"
echo "All files have been moved successfully."
