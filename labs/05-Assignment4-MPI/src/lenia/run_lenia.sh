#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=lenia
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=1
#SBATCH --output=logs/%x_%j.log
#SBATCH --hint=nomultithread

#Load MPI module 
module load OpenMPI

#Build
make

#Run
mpirun -np $SLURM_NTASKS -x FI_PROVIDER=tcp ./lenia.out

