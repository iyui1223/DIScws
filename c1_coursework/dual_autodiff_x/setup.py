from setuptools import setup, Extension
from Cython.Build import cythonize
from pathlib import Path

# Get the version from version.py
version_path = Path(__file__).parent.parent / "version.py"
version_globals = {}
with open(version_path) as f:
    exec(f.read(), version_globals)

ext_modules = cythonize(
    [
        Extension(
            "dual_autodiff_x.dual",
            sources=["dual_autodiff_x/dual.pyx"],
        )
    ]
)

setup(
    name="dual_autodiff_x",
    version=version_globals["__version__"],
    description="Cythonized dual number-based package.",
    ext_modules=ext_modules,
    packages=["dual_autodiff_x"],
    python_requires=">=3.6",
)
