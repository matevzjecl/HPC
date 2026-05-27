#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --partition=gpu
#SBATCH --job-name=lennard-jones
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=2
#SBATCH --nodes=1
#SBATCH --output=logs/%x-%j.out

# LOAD MODULES
module load CUDA

# BUILD
make

# RUN
RUNS=5
sum=0

for run in $(seq 1 "$RUNS"); do
    echo "===== Run $run / $RUNS ====="

    output="$(srun ./lj.out)"
    echo "$output"

    time_s="$(
        echo "$output" | awk '
            /Simulation time/ {
                for (i = 1; i <= NF; i++) {
                    if ($i ~ /^[0-9]+([.][0-9]+)?$/) {
                        val = $i
                    }
                }
            }
            END {
                print val
            }
        '
    )"

    echo "Run $run time: $time_s seconds"
    echo

    sum="$(awk -v sum="$sum" -v t="$time_s" 'BEGIN { printf "%.6f", sum + t }')"
done

avg="$(awk -v sum="$sum" -v runs="$RUNS" 'BEGIN { printf "%.3f", sum / runs }')"

echo "===== Average ====="
echo "Average time over $RUNS runs: $avg seconds"