#!bin/bash

# This readme provide step by step unit testings.

ROOT="/home/yi260/gitlab/DIScws/c1_coursework"
SOURCE="/home/yi260/venvs/c1coursework/bin/activate"
TESTDIR="${ROOT}/tests"

# source the compatible python environment.
source ${SOURCE}

# to export packages
cd ${ROOT}
export PYTHONPATH=${ROOT}

echo "*****************************************************"
echo "   Q1: Create project repository structure"
echo ""
cd ${ROOT}
tree -L 3

echo "*****************************************************"
echo "   Q2: Write project configuration as pyproject.toml"
echo ""
cat ${ROOT}/pyproject.toml

echo "*****************************************************"
echo "  Q3: Implement dual numbers and operations"
echo ""

echo "*****************************************************"
echo "   Q4: Make the code into a package"
echo ""

echo "*****************************************************"
echo "   Q5: Compare differentials computed by dual number, forward, and analytical method."
echo ""
python ${TESTDIR}/q5_test.py

echo "*****************************************************"
echo "   Q6: A test suite which tests the functionality of all available methods and operations of class Dual."
echo ""
pytest ${TESTDIR}/q6_test.py

echo "*****************************************************"
echo "   Q7: Write project documentation with Sphinx."
echo ""
# make sure added modules are included in modules.rst
# make html





