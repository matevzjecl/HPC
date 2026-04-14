#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --partition=gpu
#SBATCH --job-name=lenia_cuda
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --output=logs/%x_%j.log

set -euo pipefail

# LOAD MODULES
module load CUDA

# BUILD
make

# RUN 5 times and average the reported execution time
runs=5
sum=0

for i in $(seq 1 $runs); do
    echo "===== Run $i / $runs ====="

    # Capture program output
    run_output=$(srun ./lenia.out)

    # Print full output to log
    echo "$run_output"

    # Extract the numeric value from: Execution time: 20.313
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
echo "Runs: $runs"
echo "Average execution time: $avg"