#!/bin/bash
# Step-by-Step Unit Testing Guide for Q1-Q5
: '
This document provides a step-by-step guide to testing and verifying tasks from Q1-Q10. To run this as a script, simply execute it in a compatible Bash environment.
$ cd "root/directory/of/project"
$ bash README.md

## Prerequisites
A compatible Python environment is set up at CSD3: '/home/yi260/venvs/m1coursework/bin/activate'
The root directory for the project is located at CSD3: '/home/yi260/gitlab/DIScws/m1_coursework'
Alternatively .requirements_m1courswork.txt is provided, too.
'
# Define root and test directories
ROOT="/home/yi260/gitlab/DIScws/m1_coursework"
SOURCE="/home/yi260/venvs/m1coursework/bin/activate"
TESTDIR="${ROOT}/tests"

# Source the compatible Python environment
echo "Activating Python environment..."
source ${SOURCE}

# Export the project directory for Python imports
# echo "Setting PYTHONPATH..."
# export PYTHONPATH=${ROOT}

cd ${ROOT}

echo "*****************************************************"
echo "   Q1: Create dataset loader for combined images."
echo ""
python {TESTDIR}/construct_dataset.py

echo "*****************************************************"
echo "   Q2-1: Constructing fully connected neural network."
echo "Fully connected neural network was implemented as class DenseNet within Q1-2/Q2_tuning.py"
echo ""

echo "*****************************************************"
echo "   Q2-2: Hyperparameter tuning."
echo "Implemented as Q1-2/Q2_tuning.py"
echo "To run the tuning cycle, type python ${ROOT}/Q1-2/Q2_tuning.py"
echo "This may take about one hour in CSD3 icelake."
echo "Output from previous run is saved as ${ROOT}/optuna_study_results.csv"

echo "*****************************************************"
echo "  Q3: Test multiple algorithms"
echo ""
python ${ROOT}/Q3/main.py

echo "*****************************************************"
echo "   Q4: Image input comparison on classifier performance"
echo ""
python ${ROOT}/Q4/main.py

echo "*****************************************************"
echo "   Q5: Apply t-SNE analysis to input and hidden layer feature output."
echo ""
python ${ROOT}/Q5/main.py

echo "*****************************************************"
echo "   All Steps Completed Successfully!"
echo "*****************************************************"
