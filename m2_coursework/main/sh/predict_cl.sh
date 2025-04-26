#!/bin/bash

# Hyperparameter search space
learning_rates=("1e-05" "5e-05" "1e-04" )
lora_ranks=(2 4 8)

# Fixed experiment tag
#exp_name="lora-3x3"
learning_rate="1e-05"
lora_rank="2"
for context_length in "768"  "128" "512" ; do
#for learning_rate in "${learning_rates[@]}"; do
#    for lora_rank in "${lora_ranks[@]}"; do

        job_name="pred_r${lora_rank}_cl${context_length}"  

        cat > temp.sh << EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=/rds/user/yi260/hpc-work/cwk/cwssubmit/m2/yi260/main/log/${job_name}_output.log
#SBATCH --error=/rds/user/yi260/hpc-work/cwk/cwssubmit/m2/yi260/main/log/${job_name}_error.log
#SBATCH --time=02:00:00                # Max execution time (HH:MM:SS)
#SBATCH --partition=cclake
#SBATCH --cpus-per-task=16               # Request CPU cores
#SBATCH -A MPHIL-DIS-SL2-CPU            # Your project account


set -eox
source "/usr/local/software/archive/linux-scientific7-x86_64/gcc-9/miniconda3-4.7.12.1-rmuek6r3f6p3v6fdj7o2klyzta3qhslh/etc/profile.d/conda.sh"
conda init bash
conda activate m2coursework

ROOT_DIR="/rds/user/yi260/hpc-work/cwk/cwssubmit/m2/yi260/main"
exp_name="lora_r${lora_rank}_lr${learning_rate}_s5000_cl${context_length}"

WORKDIR="\${ROOT_DIR}/workdir"
SRC_DIR="\${ROOT_DIR}/src"
DATA_DIR="\${ROOT_DIR}/data"
MODELS_DIR="\${ROOT_DIR}/models"
PREDICTIONS_DIR="\${DATA_DIR}/predictions/\${exp_name}"
FIGS_DIR="\${ROOT_DIR}/figs/predictions/\${exp_name}"
LOG_DIR="\${ROOT_DIR}/log"

mkdir -p "\$WORKDIR" "\$LOG_DIR" "\$PREDICTIONS_DIR" "\$FIGS_DIR"
cd "\$WORKDIR"

echo "Setting up working directory..."

ln -sf "\${MODELS_DIR}/\${exp_name}.pt"  "\$WORKDIR/"
ln -sf "\${SRC_DIR}/"*.py "\$WORKDIR/"
ln -sf "\${DATA_DIR}/lotka_volterra_data.h5" "\$WORKDIR/"
ln -sf "\${DATA_DIR}/predictions" "\$WORKDIR/"
echo "Running batch prediction process..."

python3 predict.py --exp_name "\${exp_name}"

echo "Batch prediction completed."

echo "Organizing output files..."


echo "All files have been moved successfully."

echo "Training complete."
EOF

        # Submit job
        sbatch temp.sh
#        sh temp.sh
        break
#    done
#    break
done
