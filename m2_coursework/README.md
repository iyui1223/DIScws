# M2 Coursework Project

This repository contains code, scripts, and data for a machine learning workflow focused on predicting the dynamics of Lotka-Volterra (predator-prey) systems using a tokenized time-series approach with the Qwen2.5 model and LoRA fine-tuning.

---

## Project Structure

main/
├── data
│   ├── lotka_volterra_data.h5 # original data file.
│   ├── predator99value.npy # constant value for data compression/tokenization
│   ├── predictions # Directory for prediction output.
│   └── prey99value.npy  # constant value for data compression/tokenization
├── figs
│   ├── evaluations # figures used for evaluationg prediction
│   ├── input_stats # figures for statistical analysis of the input data
│   ├── quick_viewer.html # viewer tool
│   └── readme # usage manual for viewr tool
├── log # log output location
├── models # directory for model checkpoints -- saves only the LoRA weights
│   ├── lora_r2_lr1e-04_s1000_cl512.pt
│   ├── lora_r2_lr1e-05_s1000_cl128.pt
│   ├── lora_r2_lr1e-05_s1000_cl512.pt
│   ├── lora_r2_lr1e-05_s1000_cl768.pt
│   ├── lora_r2_lr1e-05_s5000_cl768.pt
├── pyproject.toml # .toml file for environment setup 
├── sh  # for running batch scripts for working directory setup  
│   ├── evaluate_3x3.sh # wrapper shells for submitting slurm jobs
│   ├── evaluate_cl.sh # wrapper shells for submitting slurm jobs
│   ├── evaluate.sh
│   ├── evaluate.slurm -> evaluate.sh
│   ├── lora_train_3x3.sh # wrapper shells for submitting slurm jobs
│   ├── lora_train_cl.sh # wrapper shells for submitting slurm jobs
│   ├── lora_train.sh # wrapper shells for submitting slurm jobs
│   ├── predict_3x3.sh # wrapper shells for submitting slurm jobs
│   ├── predict_cl.sh # wrapper shells for submitting slurm jobs
│   ├── predict.sh
│   ├── predict.slurm -> predict.sh
├── src
│   ├── evaluate.py # for calculating metrics for predictions
│   ├── flops.py # for flops calculation for wrapper shell
│   ├── lora_skeleton.py # the code of same name overwholed to train/resume training from user specified parameters
│   ├── plot_stat.py # for producing statistical descriptions and figures of input data
│   ├── predict.py # for predicting model 
│   ├── preprocessor.py # pre/post processor for converting numerical time series for Qwen2.5 model input tokenization.
│   └── qwen.py # For loading Qwen2.5 model
└── workdir # working directory

##  Getting Started

1. **Clone the Repository**
   ```bash
   git clone https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/m2_coursework/yi260
   cd main
   Set Up the Environment

2. **Set Up the Environment**
Create an environment using pyproject.toml

3. **Train Model**
Edit and submit relevant training scripts in sh/ such as lora_train.sh.
Check models/ for saved LoRA weight checkpoints.

4. **Make Predictions**
Submit a prediction job using predict.sh or its variants.
Outputs will be saved in data/predictions.

5. **Evaluate**
Use src/evaluate.py to compare predictions against ground truth and generate metrics and plots. 
Output will be saved at figs/ directory.


