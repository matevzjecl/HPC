#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --partition=gpu
#SBATCH --job-name=lennard-jones
#SBATCH --ntasks=1
#SBATCH --hint=nomultithread
#SBATCH --gpus-per-node=2
#SBATCH --nodes=1
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

# LOAD MODULES
module load CUDA


echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not_set}"

# BUILD
make clean
make

# RUN
RUNS=5
sum=0
valid_runs=0

for run in $(seq 1 "$RUNS"); do
    echo "===== Run $run / $RUNS ====="

    output="$(srun ./lj.out)"
    echo "$output"

    time_s="$(
        echo "$output" | awk '
            /Simulation time/ || /Execution time/ || /execution time/ || /Elapsed time/ || /elapsed time/ {
                for (i = 1; i <= NF; i++) {
                    if ($i ~ /^[-+]?[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$/) {
                        val = $i
                    }
                }
            }
            END {
                print val
            }
        '
    )"

    if [[ -z "$time_s" ]]; then
        echo "Could not parse simulation time for run $run"
        exit 1
    fi

    echo "Run $run time: $time_s seconds"
    echo

    sum="$(awk -v sum="$sum" -v t="$time_s" 'BEGIN { printf "%.6f", sum + t }')"
    valid_runs=$((valid_runs + 1))
done

avg="$(awk -v sum="$sum" -v runs="$valid_runs" 'BEGIN { printf "%.3f", sum / runs }')"

echo "===== Average ====="
echo "Average time over $valid_runs runs: $avg seconds"