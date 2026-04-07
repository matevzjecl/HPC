#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=lenia
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/%x_%j.log
#SBATCH --hint=nomultithread

# Set OpenMP environment variables for thread placement and binding    
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Load the numactl module to enable numa library linking
module load numactl

# Compile
gcc -O3 -lm -lnuma --openmp src/*.c -o lenia

# Run
srun ./lenia
