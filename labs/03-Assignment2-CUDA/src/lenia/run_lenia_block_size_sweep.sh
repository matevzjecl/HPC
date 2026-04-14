#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --partition=gpu
#SBATCH --job-name=lenia_cuda_sweep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --output=logs/%x_%j.log

set -euo pipefail

mkdir -p logs

# LOAD MODULES
module load CUDA

# BUILD
make clean
make

# Sweep square CUDA block sizes: bs x bs
block_sizes=(4 8 12 16 20 24 28 32)

# RUN each block size 5 times and average the reported execution time
runs=5

best_bs=""
best_avg=""

results_file="block_sweep_results.csv"
echo "block_size,average_time" > "$results_file"

for bs in "${block_sizes[@]}"; do
    # Safety check for square blocks
    if (( bs * bs > 1024 )); then
        echo "Skipping invalid block size ${bs}x${bs} (${bs}*${bs} > 1024 threads)"
        continue
    fi

    echo "========================================"
    echo "Testing block size: ${bs}x${bs}"
    echo "========================================"

    sum=0

    for i in $(seq 1 $runs); do
        echo "===== Run $i / $runs for block size ${bs}x${bs} ====="

        # Pass block size as argv[1]
        run_output=$(srun ./lenia.out "$bs")

        # Print full output to log
        echo "$run_output"

        # Extract the numeric value from: Execution time: 20.313
        time_val=$(echo "$run_output" | awk -F': ' '/Execution time:/ {print $2}' | tail -n 1)

        if [ -z "$time_val" ]; then
            echo "Could not find 'Execution time:' in program output for block size $bs."
            exit 1
        fi

        echo "Parsed execution time for run $i: $time_val"

        sum=$(awk -v s="$sum" -v t="$time_val" 'BEGIN {printf "%.6f", s + t}')
    done

    avg=$(awk -v s="$sum" -v r="$runs" 'BEGIN {printf "%.6f", s / r}')

    echo "Average execution time for ${bs}x${bs}: $avg"
    echo "${bs},${avg}" | tee -a "$results_file"

    if [ -z "$best_bs" ] || awk -v a="$avg" -v b="$best_avg" 'BEGIN {exit !(a < b)}'; then
        best_bs="$bs"
        best_avg="$avg"
    fi
done

echo "===== Sweep Summary ====="
cat "$results_file"
echo "Best block size: ${best_bs}x${best_bs}"
echo "Best average execution time: $best_avg"