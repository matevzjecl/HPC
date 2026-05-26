#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=lenia
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1
#SBATCH --output=logs/%x_%j.log
#SBATCH --hint=nomultithread

#Load MPI module 
module load OpenMPI

#Build
make

#Run
mpirun --mca pml ob1 -np $SLURM_NTASKS ./lenia.out

