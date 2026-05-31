#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --partition=gpu
#SBATCH --job-name=lj-openmp-sweep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --hint=nomultithread
#SBATCH --gpus-per-node=2
#SBATCH --nodes=1
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

# LOAD MODULES
module load CUDA

mkdir -p logs results

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not_set}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-not_set}"

# OPENMP BASE SETTINGS
export OMP_PROC_BIND=true
export OMP_PLACES=cores
export OMP_DYNAMIC=false

# BUILD ONCE
make clean
make

# SWEEP SETTINGS
RUNS=5

THREAD_LIST=(1 2 4 6 8 12)

SCHEDULE_LIST=(
    "static"
    "static,64"
    "static,256"
    "static,1024"
    "dynamic,64"
    "dynamic,256"
    "dynamic,1024"
    "guided,64"
    "guided,256"
    "guided,1024"
)

RESULTS_FILE="results/openmp_sweep_${SLURM_JOB_ID:-manual}.csv"

echo "threads,schedule,runs,avg_time_s,min_time_s,max_time_s" > "$RESULTS_FILE"

echo
echo "Results file: $RESULTS_FILE"
echo

for threads in "${THREAD_LIST[@]}"; do
    if (( threads > SLURM_CPUS_PER_TASK )); then
        echo "Skipping OMP_NUM_THREADS=$threads because SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
        continue
    fi

    export OMP_NUM_THREADS="$threads"

    for schedule in "${SCHEDULE_LIST[@]}"; do
        export OMP_SCHEDULE="$schedule"

        echo "============================================================"
        echo "Testing OMP_NUM_THREADS=$OMP_NUM_THREADS OMP_SCHEDULE=$OMP_SCHEDULE"
        echo "============================================================"

        sum=0
        valid_runs=0
        min_time=""
        max_time=""

        for run in $(seq 1 "$RUNS"); do
            echo "===== Run $run / $RUNS ====="

            output="$(srun --ntasks=1 --cpus-per-task="$threads" --cpu-bind=cores ./lj.out)"
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

            if [[ -z "$min_time" ]]; then
                min_time="$time_s"
                max_time="$time_s"
            else
                min_time="$(awk -v a="$min_time" -v b="$time_s" 'BEGIN { printf "%.6f", (a < b ? a : b) }')"
                max_time="$(awk -v a="$max_time" -v b="$time_s" 'BEGIN { printf "%.6f", (a > b ? a : b) }')"
            fi

            valid_runs=$((valid_runs + 1))
        done

        avg="$(awk -v sum="$sum" -v runs="$valid_runs" 'BEGIN { printf "%.6f", sum / runs }')"

        echo "===== Average for OMP_NUM_THREADS=$OMP_NUM_THREADS OMP_SCHEDULE=$OMP_SCHEDULE ====="
        echo "Average time over $valid_runs runs: $avg seconds"
        echo "Min time: $min_time seconds"
        echo "Max time: $max_time seconds"
        echo

        printf "%d,\"%s\",%d,%.6f,%.6f,%.6f\n" \
            "$threads" \
            "$schedule" \
            "$valid_runs" \
            "$avg" \
            "$min_time" \
            "$max_time" >> "$RESULTS_FILE"
    done
done

echo "============================================================"
echo "Final OpenMP sweep table"
echo "============================================================"
cat "$RESULTS_FILE"