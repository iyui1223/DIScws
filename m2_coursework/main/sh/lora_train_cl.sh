#!/bin/bash

# Hyperparameter search space
learning_rate="1e-05" # change training rate
lora_rank=2
steps=1000
isteps=1000

for context_length in "512" "128" "768"; do

    # Create valid job name by replacing '.' with 'p'
    sanitized_lr="${learning_rate//./p}"
    exp_name="lora_r${lora_rank}_lr${learning_rate}_s${steps}_cl${context_length}"
    job_name="qwen_r${lora_rank}_lr${sanitized_lr}"
    inherit_exp="lora_r${lora_rank}_lr1e-04_s${isteps}_cl512"

    temp_script="temp_${job_name}.sh"

    cat > "$temp_script" << EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=/rds/user/yi260/hpc-work/cwk/cwssubmit/m2/yi260/main/log/${job_name}_output.log
#SBATCH --error=/rds/user/yi260/hpc-work/cwk/cwssubmit/m2/yi260/main/log/${job_name}_error.log
#SBATCH --time=12:00:00                # Max execution time (HH:MM:SS)
#SBATCH --partition=icelake-himem
#SBATCH --cpus-per-task=32               # Request CPU cores
#SBATCH -A MPHIL-DIS-SL2-CPU            # Your project account

set -eox

source "/usr/local/software/archive/linux-scientific7-x86_64/gcc-9/miniconda3-4.7.12.1-rmuek6r3f6p3v6fdj7o2klyzta3qhslh/etc/profile.d/conda.sh"
conda init bash
conda activate m2coursework

ROOT_DIR="/rds/user/yi260/hpc-work/cwk/cwssubmit/m2/yi260/main"
WORKDIR="\${ROOT_DIR}/workdir/${sanitized_lr}_${lora_rank}"
SRC_DIR="\${ROOT_DIR}/src"
DATA_DIR="\${ROOT_DIR}/data"
MODELS_DIR="\${ROOT_DIR}/models"
LOG_DIR="\${ROOT_DIR}/log"

mkdir -p "\$WORKDIR" "\$LOG_DIR" "\$MODELS_DIR"
cd "\$WORKDIR"

ln -sf "\${SRC_DIR}/"*.py .
ln -sf "\${DATA_DIR}/lotka_volterra_data.h5" .
ln -sf "\${DATA_DIR}/predictions" .
ln -sf "\${MODELS_DIR}/${inherit_exp}.pt" . || echo "No existing checkpoint — starting fresh."

echo "Running training with lr=${learning_rate}, rank=${lora_rank}"

python3 lora_skeleton.py \\
--learning_rate ${learning_rate} \\
--lora_rank ${lora_rank} \\
--max_steps ${steps} \\
--context_length ${context_length} \\
--resume_training \\
--inherit_exp ${inherit_exp}
mv -v *.pt "\$MODELS_DIR/"

echo "Training complete."
EOF

    # Submit the job and remove the temp script
    sbatch "$temp_script"
#        sh "$temp_script"
    rm "$temp_script"
#        break
done
