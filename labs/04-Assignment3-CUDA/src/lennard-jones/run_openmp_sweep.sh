#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=lennard-jones_omp_sweep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --nodes=1
#SBATCH --hint=nomultithread
#SBATCH --output=logs/%x_%j.log

set -euo pipefail

module load numactl

mkdir -p logs results

RESULT_SECTION="compute_force_cells2"

N_VALUES=(1000 2000 4000 8000)
NSTEPS=5000

R_SKIN_VALUES=(
    "0.1"
    "0.2"
    "0.4"
    "0.8"
    "1.2"
)

THREAD_COUNTS=(16 32 64)

SCHED_VALUES=(
    "static"
    "static,64"
    "static,128"
    "guided,64"
)

SCHED_LABELS=(
    "static"
    "static, chunk=64"
    "static, chunk=128"
    "guided, chunk=64"
)
runs=5

export OMP_PLACES=cores
export OMP_PROC_BIND=close

EXE="lennard-jones_${SLURM_JOB_ID}"
RESULT_FILE="results/sweep_${RESULT_SECTION}_${SLURM_JOB_ID}.md"

cleanup() {
    rm -f "$EXE"
}
trap cleanup EXIT

gcc -O3 -fopenmp src/*.c -o "$EXE" -lm -lnuma

echo "Writing results to: $RESULT_FILE"
echo

{
    echo "## ${RESULT_SECTION}:"
    echo "    - nsteps: ${NSTEPS}"
} > "$RESULT_FILE"

for n_value in "${N_VALUES[@]}"; do
    {
        echo "    - n=${n_value}:"
    } >> "$RESULT_FILE"

    echo
    echo "##################################################"
    echo "N = ${n_value}"
    echo "nsteps = ${NSTEPS}"
    echo "##################################################"

    for r_skin_value in "${R_SKIN_VALUES[@]}"; do
        {
            echo "        - r_skin=${r_skin_value}:"
        } >> "$RESULT_FILE"

        echo
        echo "=================================================="
        echo "N: ${n_value}"
        echo "nsteps: ${NSTEPS}"
        echo "r_skin: ${r_skin_value}"
        echo "=================================================="

        for si in "${!SCHED_VALUES[@]}"; do
            sched_value="${SCHED_VALUES[$si]}"
            sched_label="${SCHED_LABELS[$si]}"

            {
                echo "            - ${sched_label}:"
            } >> "$RESULT_FILE"

            echo
            echo "========================================"
            echo "N: ${n_value}"
            echo "nsteps: ${NSTEPS}"
            echo "r_skin: ${r_skin_value}"
            echo "Schedule: ${sched_label}"
            echo "OMP_SCHEDULE=${sched_value}"
            echo "========================================"

            for threads in "${THREAD_COUNTS[@]}"; do
                export OMP_NUM_THREADS="$threads"
                export OMP_SCHEDULE="$sched_value"

                sum=0

                echo
                echo "----------------------------------------"
                echo "N: $n_value"
                echo "nsteps: $NSTEPS"
                echo "r_skin: $r_skin_value"
                echo "Threads: $threads"
                echo "Schedule: $sched_label"
                echo "----------------------------------------"

                for i in $(seq 1 "$runs"); do
                    echo "===== Run $i / $runs ====="

                    run_output=$(srun --ntasks=1 --cpus-per-task="$threads" ./"$EXE" "$n_value" "$NSTEPS" "$r_skin_value")

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

                echo "Average for n=$n_value nsteps=$NSTEPS r_skin=$r_skin_value threads=$threads schedule=$sched_label: $avg"

                {
                    echo "                - ${threads}: ${avg}"
                } >> "$RESULT_FILE"
            done
        done
    done
done

echo
echo "===== Final sweep result ====="
cat "$RESULT_FILE"