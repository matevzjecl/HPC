#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=lenia_omp_sched
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --hint=nomultithread
#SBATCH --output=logs/%x_%j.log

set -euo pipefail

mkdir -p logs

# LOAD MODULES
module load numactl

# OpenMP settings
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# BUILD
gcc -O3 -lm -lnuma --openmp src/*.c -o lenia

# Schedules to test
schedules=(
    "static"
    "static,1"
    "static,4"
    "dynamic,1"
    "dynamic,4"
    "guided,1"
    "guided,4"
)

runs=5

echo "===== OpenMP schedule sweep ====="
echo "Threads: $OMP_NUM_THREADS"
echo "Runs per schedule: $runs"
echo

for sched in "${schedules[@]}"; do
    export OMP_SCHEDULE="$sched"

    echo "========================================"
    echo "Testing schedule: $OMP_SCHEDULE"
    echo "========================================"

    sum=0

    for i in $(seq 1 $runs); do
        echo "----- Run $i / $runs -----"

        run_output=$(srun ./lenia)

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

    echo "----- Schedule summary -----"
    echo "OMP_SCHEDULE: $OMP_SCHEDULE"
    echo "Average execution time: $avg"
    echo
done