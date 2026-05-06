#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --partition=gpu
#SBATCH --job-name=lennard-jones_cuda
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --output=logs/%x_%j.log

set -euo pipefail

module load CUDA

mkdir -p logs results

make

warmups=3
runs=5
sum=0

echo "===== Warm-up runs ====="
for i in $(seq 1 "$warmups"); do
    echo "===== Warm-up $i / $warmups ====="
    srun ./lj.out > /dev/null
done

echo "===== Timed runs ====="
for i in $(seq 1 "$runs"); do
    echo "===== Run $i / $runs ====="

    run_output=$(srun ./lj.out)

    echo "$run_output"

    time_val=$(echo "$run_output" | awk '/Simulation time/ {print $(NF-1)}' | tail -n 1)

    if [ -z "$time_val" ]; then
        echo "Could not find simulation time in program output."
        exit 1
    fi

    echo "Parsed simulation time for run $i: $time_val"

    sum=$(awk -v s="$sum" -v t="$time_val" 'BEGIN {printf "%.6f", s + t}')
done

avg=$(awk -v s="$sum" -v r="$runs" 'BEGIN {printf "%.6f", s / r}')

echo "===== Summary ====="
echo "Warmups: $warmups"
echo "Runs: $runs"
echo "Average simulation time: $avg"