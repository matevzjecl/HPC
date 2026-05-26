#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=lenia_sweep
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1
#SBATCH --output=logs/%x_%j.log
#SBATCH --hint=nomultithread

set -euo pipefail

module load OpenMPI

mkdir -p logs results

GRID_SIZES=(128 512 1024 2048 4096)
CORE_COUNTS=(1 2 4 16 32)

RUNS=5

JOB_ID="${SLURM_JOB_ID:-manual}"
RESULT_FILE="results/lenia_sweep_${JOB_ID}.csv"

echo "===== Building Lenia ====="
make clean
make

echo "grid_size,cores,steps,runs,avg_time" > "$RESULT_FILE"

echo "===== Lenia MPI sweep ====="
echo "Grid sizes: ${GRID_SIZES[*]}"
echo "Core counts: ${CORE_COUNTS[*]}"
echo "Runs per configuration: $RUNS"
echo "Result file: $RESULT_FILE"
echo

for N in "${GRID_SIZES[@]}"; do
    for CORES in "${CORE_COUNTS[@]}"; do
        echo "===== Running N=$N with $CORES cores ====="

        total_time=0

        for RUN in $(seq 1 "$RUNS"); do
            RUN_LOG="logs/lenia_N${N}_p${CORES}_run${RUN}_${JOB_ID}.log"

            echo "Run $RUN / $RUNS"

            mpirun --mca pml ob1 -np "$CORES" ./lenia.out "$N" > "$RUN_LOG" 2>&1

            # Use the maximum reported rank time.
            # This is the correct MPI runtime because the whole program is limited by the slowest rank.
            sim_time=$(awk '
                /Execution time:/ {
                    if ($3 > max) max = $3
                    found = 1
                }
                END {
                    if (!found) exit 1
                    printf "%.6f", max
                }
            ' "$RUN_LOG") || {
                echo "ERROR: Could not find execution time in $RUN_LOG"
                echo "Last lines of log:"
                tail -40 "$RUN_LOG"
                exit 1
            }

            total_time=$(awk -v total="$total_time" -v t="$sim_time" 'BEGIN { printf "%.6f", total + t }')

            echo "N=$N cores=$CORES run=$RUN time=$sim_time"
        done

        avg_time=$(awk -v total="$total_time" -v runs="$RUNS" 'BEGIN { printf "%.6f", total / runs }')

        echo "$N,$CORES,100,$RUNS,$avg_time" >> "$RESULT_FILE"

        echo "Average: N=$N cores=$CORES avg_time=$avg_time"
        echo
    done
done

echo "===== Sweep finished ====="
echo "Results saved to: $RESULT_FILE"