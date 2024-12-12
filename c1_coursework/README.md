#!/bin/bash
# Step-by-Step Unit Testing Guide for Q1-Q10
: '
This document provides a step-by-step guide to testing and verifying tasks from Q1-Q10. To run this as a script, simply execute it in a compatible Bash environment.
$ cd "root/directory/of/project"
$ bash README.md

## Prerequisites
A compatible Python environment is set up at CSD3: '/home/yi260/venvs/c1coursework/bin/activate'
The root directory for the project is located at CSD3: '/home/yi260/gitlab/DIScws/c1_coursework'

'
# Define root and test directories
ROOT="/home/yi260/gitlab/DIScws/c1_coursework"
SOURCE="/home/yi260/venvs/c1coursework/bin/activate"
TESTDIR="${ROOT}/tests"

# Source the compatible Python environment
echo "Activating Python environment..."
source ${SOURCE}

# Export the project directory for Python imports
echo "Setting PYTHONPATH..."
export PYTHONPATH=${ROOT}

# Navigate to the project root
cd ${ROOT}

echo "*****************************************************"
echo "   Q1: Create project repository structure"
echo ""
echo "Verify the project directory structure:"
tree -L 3 ${ROOT}

echo "*****************************************************"
echo "   Q2: Write project configuration as pyproject.toml"
echo ""
echo "Display the pyproject.toml content:"
cat ${ROOT}/pyproject.toml

echo "Installing the project in editable mode:"
pip install -e .

echo "*****************************************************"
echo "  Q3: Implement dual numbers and operations"
echo ""
echo "Verified as the successful execution of test for Q5."

echo "*****************************************************"
echo "   Q4: Make the code into a package"
echo ""
echo "Verified as the successful execution of test for Q5."

echo "*****************************************************"
echo "   Q5: Compare differentials computed by dual number, forward, and analytical method."
echo ""
echo "Running comparison script:"
python ${TESTDIR}/q5_test.py

echo "*****************************************************"
echo "   Q6: A test suite which tests the functionality of all available methods and operations of class Dual."
echo ""
echo "Running pytest for Dual class functionality:"
pytest ${TESTDIR}/q6_test.py

echo "*****************************************************"
echo "   Q7: Write project documentation with Sphinx."
echo ""
echo "Generating documentation:"
cd ${ROOT}
make clean
make html

echo "*****************************************************"
echo "   Q8: Cythonize the package."
echo ""
echo "Installing the Cythonized package:"
cd ${ROOT}/dual_autodiff_x
pip install -e .

echo "*****************************************************"
echo "   Q9: Performance compalison."
echo ""
echo "The performance report is provided as tests/dual_autodiff.ipynb"

echo "*****************************************************"
echo "   Q10: Build wheel for distribution."
echo ""
cd ${ROOT}
rm -rf "${ROOT}/build" "${ROOT}/dist" "${ROOT}/*.egg-info"
# Search pre-installed docker.
# module avail | grep "docker"
# load the latest available docker.
module load ceuadmin/docker/27.0.3
cibuildwheel --output-dir dist

echo "*****************************************************"
echo "   All Steps Completed Successfully!"
echo "*****************************************************"
