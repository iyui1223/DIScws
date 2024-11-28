#!bin/bash

# This readme provide step by step unit testings.

ROOT="/home/yi260/gitlab/DIScws/c1_coursework"
SOURCE="/home/yi260/venvs/c1coursework/bin/activate"
TESTDIR="${ROOT}/tests"

# source the compatible python environment.
source ${SOURCE}

# to export packages
export PYTHONPATH=$(pwd)

# Q1: Create project repository structure
# cd ${ROOT}
# tree -L 3

# Q2: Write project configuration as pyproject.toml

# Q3: Implement dual numbers and operations

# Q4: Make the code into a package

# Q5: Compare differentials computed by dual number, forward differential, and analytical method.
# python ${TESTDIR}/q5_test.py

# Q6: A test suite which tests the functionality of all available methods and operations of class Dual.
# python ${TESTDIR}/q6_test.py






