#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --partition=gpu
#SBATCH --job-name=lj_param_sweep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --hint=nomultithread
#SBATCH --gpus-per-node=2
#SBATCH --nodes=1
#SBATCH --output=lj_param_sweep-%j.out

set -uo pipefail

module load CUDA

# OpenMP settings
# Keep this at 6 because your previous tests/scripts used 6 even though 12 CPUs are allocated.
# Override with: OMP_THREADS=12 sbatch script.sh
export OMP_NUM_THREADS="${OMP_THREADS:-6}"
export OMP_PROC_BIND=true
export OMP_PLACES=cores
export OMP_SCHEDULE="static"

mkdir -p logs
mkdir -p results/raw

RUNS="${RUNS:-5}"
CHANGE_TOL="${CHANGE_TOL:-0.000050}"
RUN_TIMEOUT="${RUN_TIMEOUT:-120}"

CSV="results/lj_param_sweep.csv"

BASE_CFLAGS="-Wno-deprecated-gpu-targets -gencode=arch=compute_70,code=sm_70 -O3 --use_fast_math -Xcompiler=-fopenmp,-Wall"

echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "OMP_PROC_BIND=${OMP_PROC_BIND}"
echo "OMP_PLACES=${OMP_PLACES}"
echo "OMP_SCHEDULE=${OMP_SCHEDULE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not_set}"
echo "RUNS=${RUNS}"
echo "CHANGE_TOL=${CHANGE_TOL}"
echo "RUN_TIMEOUT=${RUN_TIMEOUT}"

echo "r_skin,max_neighbors,rebuild_interval,runs,avg_time_s,min_time_s,max_time_s,avg_change,baseline_change,avg_abs_diff,status" > "$CSV"

build_lj() {
    local max_neighbors="$1"
    local rebuild_interval="$2"

    echo
    echo "Building: LJ_MAX_NEIGHBORS=${max_neighbors}, LJ_REBUILD_INTERVAL=${rebuild_interval}"

    make clean >/dev/null 2>&1 || true

    if ! make CFLAGS="${BASE_CFLAGS} -DLJ_MAX_NEIGHBORS=${max_neighbors} -DLJ_REBUILD_INTERVAL=${rebuild_interval}"; then
        echo "ERROR: build failed"
        exit 1
    fi

    if [ ! -x ./lj.out ]; then
        echo "ERROR: ./lj.out was not created"
        exit 1
    fi
}

parse_sim_time() {
    local log_file="$1"

    awk '
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
    ' "$log_file"
}

parse_change() {
    local log_file="$1"

    awk '
        /Change/ {
            for (i = 1; i <= NF; i++) {
                if ($i ~ /^[-+]?[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$/) {
                    val = $i
                    break
                }
            }
        }
        END {
            print val
        }
    ' "$log_file"
}

run_one() {
    local r_skin="$1"
    local max_neighbors="$2"
    local rebuild_interval="$3"
    local run="$4"
    local raw_log="$5"

    echo
    echo "Running: r_skin=${r_skin}, max_neighbors=${max_neighbors}, rebuild_interval=${rebuild_interval}, run=${run}"
    echo "Log: ${raw_log}"

    export LJ_R_SKIN="$r_skin"

    local start_time
    local end_time
    local bash_elapsed
    local cmd_pid
    local rc
    local sim_time
    local change

    RUN_TIME=""
    RUN_CHANGE=""
    RUN_BASH_ELAPSED=""
    RUN_STATUS=""

    : > "$raw_log"

    start_time="$(date +%s.%N)"

    # Run in its own process group so overflow/timeout can kill only this run.
    setsid timeout --kill-after=10s "${RUN_TIMEOUT}s" ./lj.out > "$raw_log" 2>&1 < /dev/null &
    cmd_pid="$!"

    while kill -0 "$cmd_pid" 2>/dev/null; do
        if grep -q "Warning: neighbor list overflow" "$raw_log"; then
            end_time="$(date +%s.%N)"
            bash_elapsed="$(awk -v s="$start_time" -v e="$end_time" 'BEGIN { printf "%.6f", e - s }')"

            echo "WARNING: neighbor list overflow detected. Killing this run."
            echo "Parameters: r_skin=${r_skin}, max_neighbors=${max_neighbors}, rebuild_interval=${rebuild_interval}, run=${run}"
            grep -m 1 "Warning: neighbor list overflow" "$raw_log" || true

            kill -TERM "-$cmd_pid" 2>/dev/null || true
            sleep 1
            kill -KILL "-$cmd_pid" 2>/dev/null || true
            wait "$cmd_pid" 2>/dev/null || true

            RUN_BASH_ELAPSED="$bash_elapsed"
            RUN_STATUS="overflow"
            return 2
        fi

        sleep 0.2
    done

    wait "$cmd_pid"
    rc="$?"

    end_time="$(date +%s.%N)"
    bash_elapsed="$(awk -v s="$start_time" -v e="$end_time" 'BEGIN { printf "%.6f", e - s }')"
    RUN_BASH_ELAPSED="$bash_elapsed"

    if grep -q "Warning: neighbor list overflow" "$raw_log"; then
        echo "WARNING: neighbor list overflow detected after run ended."
        echo "Parameters: r_skin=${r_skin}, max_neighbors=${max_neighbors}, rebuild_interval=${rebuild_interval}, run=${run}"
        grep -m 1 "Warning: neighbor list overflow" "$raw_log" || true

        RUN_STATUS="overflow"
        return 2
    fi

    if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
        echo "WARNING: run timed out after ${RUN_TIMEOUT}s."
        echo "Parameters: r_skin=${r_skin}, max_neighbors=${max_neighbors}, rebuild_interval=${rebuild_interval}, run=${run}"
        tail -20 "$raw_log" || true

        RUN_STATUS="timeout"
        return 4
    fi

    if [ "$rc" -ne 0 ]; then
        echo "WARNING: lj.out failed with exit code ${rc}."
        echo "Parameters: r_skin=${r_skin}, max_neighbors=${max_neighbors}, rebuild_interval=${rebuild_interval}, run=${run}"
        tail -20 "$raw_log" || true

        RUN_STATUS="failed"
        return 1
    fi

    sim_time="$(parse_sim_time "$raw_log")"
    change="$(parse_change "$raw_log")"

    if [ -z "$sim_time" ]; then
        echo "WARNING: could not parse simulation time from log."
        echo "Parameters: r_skin=${r_skin}, max_neighbors=${max_neighbors}, rebuild_interval=${rebuild_interval}, run=${run}"
        tail -20 "$raw_log" || true

        RUN_STATUS="no_time"
        return 3
    fi

    if [ -z "$change" ]; then
        echo "WARNING: could not parse Change from log."
        echo "Parameters: r_skin=${r_skin}, max_neighbors=${max_neighbors}, rebuild_interval=${rebuild_interval}, run=${run}"
        tail -20 "$raw_log" || true

        RUN_STATUS="no_change"
        return 3
    fi

    RUN_TIME="$sim_time"
    RUN_CHANGE="$change"
    RUN_STATUS="ok"

    echo "Finished: sim_time=${RUN_TIME}, bash_elapsed=${RUN_BASH_ELAPSED}, change=${RUN_CHANGE}"
    return 0
}

write_failed_case() {
    local r_skin="$1"
    local max_neighbors="$2"
    local rebuild_interval="$3"
    local completed_runs="$4"
    local sum_time="$5"
    local min_time="$6"
    local max_time="$7"
    local sum_change="$8"
    local status="$9"

    local avg_time=""
    local avg_change=""
    local avg_abs_diff=""

    if [ "$completed_runs" -gt 0 ]; then
        avg_time="$(awk -v s="$sum_time" -v r="$completed_runs" 'BEGIN { printf "%.6f", s / r }')"
        avg_change="$(awk -v s="$sum_change" -v r="$completed_runs" 'BEGIN { printf "%.9f", s / r }')"

        avg_abs_diff="$(awk -v c="$avg_change" -v b="$BASELINE_CHANGE" 'BEGIN {
            d = c - b
            if (d < 0) d = -d
            printf "%.9f", d
        }')"
    fi

    echo "${r_skin},${max_neighbors},${rebuild_interval},${completed_runs},${avg_time},${min_time},${max_time},${avg_change},${BASELINE_CHANGE},${avg_abs_diff},${status}" >> "$CSV"
}

average_case() {
    local r_skin="$1"
    local max_neighbors="$2"
    local rebuild_interval="$3"

    local sum_time="0"
    local min_time=""
    local max_time=""
    local sum_change="0"
    local completed_runs="0"
    local raw_log
    local avg_time
    local avg_change
    local avg_abs_diff
    local status

    for run in $(seq 1 "$RUNS"); do
        raw_log="results/raw/r${r_skin}_max${max_neighbors}_rebuild${rebuild_interval}_run${run}.log"

        if run_one "$r_skin" "$max_neighbors" "$rebuild_interval" "$run" "$raw_log"; then
            completed_runs=$((completed_runs + 1))

            sum_time="$(awk -v a="$sum_time" -v b="$RUN_TIME" 'BEGIN { printf "%.9f", a + b }')"
            sum_change="$(awk -v a="$sum_change" -v b="$RUN_CHANGE" 'BEGIN { printf "%.12f", a + b }')"

            if [ -z "$min_time" ]; then
                min_time="$RUN_TIME"
                max_time="$RUN_TIME"
            else
                min_time="$(awk -v a="$min_time" -v b="$RUN_TIME" 'BEGIN { if (b < a) print b; else print a }')"
                max_time="$(awk -v a="$max_time" -v b="$RUN_TIME" 'BEGIN { if (b > a) print b; else print a }')"
            fi
        else
            echo "Skipping this parameter case after run ${run}, status=${RUN_STATUS}."
            write_failed_case "$r_skin" "$max_neighbors" "$rebuild_interval" "$completed_runs" "$sum_time" "$min_time" "$max_time" "$sum_change" "$RUN_STATUS"
            return 0
        fi
    done

    avg_time="$(awk -v s="$sum_time" -v r="$completed_runs" 'BEGIN { printf "%.6f", s / r }')"
    avg_change="$(awk -v s="$sum_change" -v r="$completed_runs" 'BEGIN { printf "%.9f", s / r }')"

    avg_abs_diff="$(awk -v c="$avg_change" -v b="$BASELINE_CHANGE" 'BEGIN {
        d = c - b
        if (d < 0) d = -d
        printf "%.9f", d
    }')"

    status="$(awk -v d="$avg_abs_diff" -v tol="$CHANGE_TOL" 'BEGIN {
        if (d > tol) print "possibly_incorrect"
        else print "ok"
    }')"

    echo "${r_skin},${max_neighbors},${rebuild_interval},${completed_runs},${avg_time},${min_time},${max_time},${avg_change},${BASELINE_CHANGE},${avg_abs_diff},${status}" >> "$CSV"

    echo "Average: r_skin=${r_skin}, max_neighbors=${max_neighbors}, rebuild_interval=${rebuild_interval}, avg_time=${avg_time}, avg_change=${avg_change}, diff=${avg_abs_diff}, status=${status}"
}

echo
echo "===== BASELINE ====="

BASELINE_R_SKIN="1.0"
BASELINE_MAX_NEIGHBORS="256"
BASELINE_REBUILD_INTERVAL="5"

build_lj "$BASELINE_MAX_NEIGHBORS" "$BASELINE_REBUILD_INTERVAL"

baseline_log="results/raw/baseline.log"

if ! run_one "$BASELINE_R_SKIN" "$BASELINE_MAX_NEIGHBORS" "$BASELINE_REBUILD_INTERVAL" "1" "$baseline_log"; then
    echo
    echo "ERROR: baseline failed with status=${RUN_STATUS}."
    echo "Baseline must succeed before sweep can be checked."
    echo "Log: ${baseline_log}"
    exit 1
fi

BASELINE_TIME="$RUN_TIME"
BASELINE_CHANGE="$RUN_CHANGE"

echo
echo "Baseline simulation time = ${BASELINE_TIME}"
echo "Baseline Change          = ${BASELINE_CHANGE}"
echo "Baseline log             = ${baseline_log}"

echo
echo "===== SWEEP ====="

R_SKINS=(
    "0.2"
    "0.3"
    "0.4"
)

MAX_NEIGHBORS_LIST=(
    "128"
    "192"
)

REBUILD_INTERVALS=(
    "15"
    "20"
    "25"
    "30"
    "35"
)

for max_neighbors in "${MAX_NEIGHBORS_LIST[@]}"; do
    for rebuild_interval in "${REBUILD_INTERVALS[@]}"; do
        build_lj "$max_neighbors" "$rebuild_interval"

        for r_skin in "${R_SKINS[@]}"; do
            average_case "$r_skin" "$max_neighbors" "$rebuild_interval"
        done
    done
done

echo
echo "Done."
echo "CSV: ${CSV}"
echo "Raw logs: results/raw/"