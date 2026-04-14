#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=lenia_omp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --hint=nomultithread
#SBATCH --output=logs/%x_%j.log

set -euo pipefail

# LOAD MODULES
module load numactl

# OpenMP settings
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# Unique executable for this job
EXE="lenia_${SLURM_JOB_ID}"

# Cleanup on exit
cleanup() {
    rm -f "$EXE"
}
trap cleanup EXIT

# BUILD
gcc -O3 -fopenmp src/*.c -o "$EXE" -lm -lnuma

# RUN 5 times and average the reported execution time
runs=5
sum=0

for i in $(seq 1 $runs); do
    echo "===== Run $i / $runs ====="

    run_output=$(srun ./"$EXE")

    echo "$run_output"

    time_val=$(echo "$run_output" | awk -F': ' '/Execution time:/ {print $2}' | tail -n 1)

    if [ -z "$time_val" ]; then
        echo "Could not find 'Execution time:' in program output."
        exit 1
    fi

    echo "Parsed execution time for run $i: $time_val"

    sum=$(awk -v s="$sum" -v t="$time_val" 'BEGIN {printf "%.6f", s + t}')
done

avg=$(awk -v s="$sum" -v r="$runs" 'BEGIN {printf "%.6f", s / r}')

echo "===== Summary ====="
echo "OpenMP threads: $OMP_NUM_THREADS"
echo "Runs: $runs"
echo "Average execution time: $avg"