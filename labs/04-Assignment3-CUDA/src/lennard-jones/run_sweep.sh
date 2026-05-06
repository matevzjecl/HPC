#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --partition=gpu
#SBATCH --job-name=lennard-jones_cuda_sweep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --output=logs/%x_%j.log

set -euo pipefail

module load CUDA

mkdir -p logs results

make

N_VALUES=(1000 2000 4000 8000)
NSTEPS=5000

R_SKIN_VALUES=(
    "0.2"
    "0.4"
    "0.6"
    "0.8"
    "1.0"
)

BLOCK_SIZES=(32 64 128 256 512 1024)

warmup_runs=3
runs=5

RESULT_FILE="results/cuda_sweep_${SLURM_JOB_ID}.csv"

echo "n,nsteps,r_skin,block_size,runs,warmup_runs,avg_time" > "$RESULT_FILE"

echo "===== CUDA Lennard-Jones sweep ====="
echo "Result file: $RESULT_FILE"
echo "Warmup runs per configuration: $warmup_runs"
echo "Measured runs per configuration: $runs"
echo

for n in "${N_VALUES[@]}"; do
    for r_skin in "${R_SKIN_VALUES[@]}"; do
        for block_size in "${BLOCK_SIZES[@]}"; do

            echo "========================================"
            echo "n=$n, nsteps=$NSTEPS, r_skin=$r_skin, block_size=$block_size"
            echo "========================================"

            echo "----- Warmup runs, not counted -----"

            for i in $(seq 1 "$warmup_runs"); do
                echo "===== Warmup $i / $warmup_runs ====="

                run_output=$(srun ./lj.out "$n" "$NSTEPS" "$r_skin" "$block_size")

                echo "$run_output"

                time_val=$(echo "$run_output" | awk -F': ' '/Simulation time / {print $2}' | tail -n 1)

                if [ -z "$time_val" ]; then
                    echo "Could not find 'Simulation time ' in warmup output."
                    exit 1
                fi

                echo "Warmup simulation time $i, not counted: $time_val"
            done

            echo "----- Measured runs -----"

            sum=0

            for i in $(seq 1 "$runs"); do
                echo "===== Run $i / $runs ====="

                run_output=$(srun ./lj.out "$n" "$NSTEPS" "$r_skin" "$block_size")

                echo "$run_output"

                time_val=$(echo "$run_output" | awk -F': ' '/Simulation time / {print $2}' | tail -n 1)

                if [ -z "$time_val" ]; then
                    echo "Could not find 'Simulation time ' in program output."
                    exit 1
                fi

                echo "Parsed simulation time for run $i: $time_val"

                sum=$(awk -v s="$sum" -v t="$time_val" 'BEGIN {printf "%.6f", s + t}')
            done

            avg=$(awk -v s="$sum" -v r="$runs" 'BEGIN {printf "%.6f", s / r}')

            echo "Average simulation time: $avg"
            echo "$n,$NSTEPS,$r_skin,$block_size,$runs,$warmup_runs,$avg" >> "$RESULT_FILE"
            echo

        done
    done
done

echo "===== Sweep complete ====="
echo "Saved results to: $RESULT_FILE"

echo
echo "===== Best results ====="
{
    head -n 1 "$RESULT_FILE"
    tail -n +2 "$RESULT_FILE" | sort -t, -k7,7n | head -n 10
}