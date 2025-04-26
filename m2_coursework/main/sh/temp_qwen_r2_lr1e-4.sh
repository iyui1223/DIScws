#!/bin/bash
#SBATCH --job-name=qwen_r2_lr1e-4
#SBATCH --output=/rds/user/yi260/hpc-work/cwk/cwssubmit/m2/yi260/main/log/qwen_r2_lr1e-4_output.log
#SBATCH --error=/rds/user/yi260/hpc-work/cwk/cwssubmit/m2/yi260/main/log/qwen_r2_lr1e-4_error.log
#SBATCH --time=00:15:00                # Max execution time (HH:MM:SS)
#SBATCH --partition=icelake-himem
#SBATCH --cpus-per-task=4               # Request CPU cores
#SBATCH -A MPHIL-DIS-SL2-CPU            # Your project account

set -eox

source "/usr/local/software/archive/linux-scientific7-x86_64/gcc-9/miniconda3-4.7.12.1-rmuek6r3f6p3v6fdj7o2klyzta3qhslh/etc/profile.d/conda.sh"
conda init bash
conda activate m2coursework

ROOT_DIR="/rds/user/yi260/hpc-work/cwk/cwssubmit/m2/yi260/main"
WORKDIR="${ROOT_DIR}/workdir/1e-4_2"
SRC_DIR="${ROOT_DIR}/src"
DATA_DIR="${ROOT_DIR}/data"
MODELS_DIR="${ROOT_DIR}/models"
LOG_DIR="${ROOT_DIR}/log"

mkdir -p "$WORKDIR" "$LOG_DIR" "$MODELS_DIR"
cd "$WORKDIR"

ln -sf "${SRC_DIR}/"*.py .
ln -sf "${DATA_DIR}/lotka_volterra_data.h5" .
ln -sf "${DATA_DIR}/predictions" .
ln -sf "${MODELS_DIR}/lora_r2_lr1e-4_s25_cl512.pt" . || echo "No existing checkpoint — starting fresh."

echo "Running training with lr=1e-4, rank=2"

python3 lora_skeleton.py \
    --learning_rate 1e-4 \
    --lora_rank 2 \
    --max_steps 25 \
    --context_length 512 \
    --resume_training \
    --inherit_exp lora_r2_lr1e-4_s25_cl512
mv -v *.pt "$MODELS_DIR/"

echo "Training complete."
