# C2 Coursework

This repository contains my solution scripts for the C2 coursework assignment. The coursework involves optimizing and parallelizing a heat diffusion simulation using various techniques including cache blocking, floating-point precision conversion, and hybrid MPI/OpenMP parallelism.

## Repository Structure
animate.ipynb # Jupyter notebook for visualizing simulation results
main/
 ├── CMakeLists.txt # CMake configuration file
 ├── Data/ # Output results from simulation runs
 ├── Log/ # SLURM job logs 
 ├── Sh/ # Job execution scripts 
 │ ├── run_non-parallel.slurm 
 │ ├── run_parallel_mpiomp.slurm 
 │ ├── run_parallel_mpi.slurm 
 │ ├── run_parallel_omp.slurm 
 └── Source/ # Source code for various implementations 
     ├── original.cpp # Baseline code
     ├── non-parallel.cpp # Adds binary output & array-style conversion 
     ├── non-parallel-blocking.cpp # Adds cache blocking to non-parallel.cpp 
     ├── non-parallel-float.cpp # Converts double to float in non-parallel.cpp 
     ├── non-parallel-fb.cpp # Combines float conversion and cache blocking 
     ├── parallel-mpi.cpp # MPI-parallelized version of non-parallel-float.cpp 
     ├── parallel-omp.cpp # OpenMP-parallelized version 
     └── parallel-mpiomp.cpp # Hybrid MPI + OpenMP version
README.md # This file


## How to Run
To run the simulations on CSD3:

Example:
```bash
cd main/Sh
# modify the ${Root} path
sbatch run_parallel_mpiomp.slurm

# The results can be examined through animate.ipynb