#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --partition=gpu
#SBATCH --job-name=lj-force-omp-graphs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --output=logs/%x-%j.out

set -u

module load CUDA

mkdir -p logs results/raw

RUNS=10

# Six benchmark-like cases: three below 50k, three above 50k, max 100k.
# Adjust nsteps here if your final benchmark uses different step counts.
CONFIGS=(
    "10000 5000"
    "50000 5000"
    "75000 5000"
    "80000 5000"
    "90000 5000"
    "100000 5000"
)

# Force block sizes must stay powers of two because the code sanitizes block sizes.
FORCE_BLOCKS=(256)
OPENMP_MODES=(0)
CUDA_GRAPH_MODES=(1)

# Keep these fixed during this sweep so the result isolates force block / OpenMP / graphs.
R_SKIN=0.3
LJ_MAX_NEIGHBORS=128
LJ_REBUILD_INTERVAL=30
LJ_DYNAMIC_REBUILD=0

OUT_RUNS="results/force_openmp_graphs_runs.csv"
OUT_SUMMARY="results/force_openmp_graphs_summary.csv"

printf "n,nsteps,force_block,openmp,cuda_graphs,run,time_s,change,status,raw_log\n" > "$OUT_RUNS"

BASE_NVCC_FLAGS="-Wno-deprecated-gpu-targets -gencode=arch=compute_70,code=sm_70 -O3 --use_fast_math -Xcompiler=-fopenmp,-Wall"

compile_variant() {
    local force_block="$1"
    local use_openmp="$2"
    local use_graphs="$3"

    local host_flags
    local omp_threads

    if [[ "$use_openmp" == "1" ]]; then
        host_flags="-Xcompiler=-fopenmp,-Wall"
        omp_threads="$SLURM_CPUS_PER_TASK"
    else
        host_flags="-Xcompiler=-Wall"
        omp_threads="1"
    fi

    local defs="-DFORCE_BLOCK_SIZE=${force_block}"
    defs+=" -DLJ_USE_OPENMP=${use_openmp}"
    defs+=" -DLJ_USE_CUDA_GRAPHS=${use_graphs}"
    defs+=" -DLJ_USE_DYNAMIC_REBUILD=${LJ_DYNAMIC_REBUILD}"
    defs+=" -DR_SKIN=${R_SKIN}f"
    defs+=" -DLJ_MAX_NEIGHBORS=${LJ_MAX_NEIGHBORS}"
    defs+=" -DLJ_REBUILD_INTERVAL=${LJ_REBUILD_INTERVAL}"
    defs+=" -DLJ_LOG_MAX_NEIGHBORS=0"

    echo "===== Building force_block=${force_block} openmp=${use_openmp} cuda_graphs=${use_graphs} ====="
    make clean >/dev/null 2>&1 || true
    make CFLAGS="${BASE_NVCC_FLAGS} ${host_flags} ${defs}"

}

run_one() {
    local n="$1"
    local nsteps="$2"
    local force_block="$3"
    local use_openmp="$4"
    local use_graphs="$5"
    local run="$6"

    local raw="results/raw/n${n}_steps${nsteps}_fb${force_block}_omp${use_openmp}_graphs${use_graphs}_run${run}.log"

    echo "Running n=${n} nsteps=${nsteps} force_block=${force_block} openmp=${use_openmp} cuda_graphs=${use_graphs} run=${run}/${RUNS}"

    set +e
    LJ_PRINT_PARAMS=1 /usr/bin/time -f "WALL_TIME %e" srun ./lj.out "$n" "$nsteps" > "$raw" 2>&1
    local rc=$?
    set -e

    local time_s
    time_s=$(awk '
        /Simulation time[[:space:]]+[0-9]+[[:space:]]+steps:/ {
            for (i = 1; i <= NF; i++) {
                if ($i == "seconds") {
                    value = $(i - 1)
                }
            }
        }
        /Execution time:/ {
            value = $NF
        }
        END {
            if (value != "") print value
        }
    ' "$raw")

    # Fallback only: this is total process wall time, not the program-reported simulation time.
    if [[ -z "$time_s" ]]; then
        time_s=$(awk '
            /WALL_TIME/ {
                value = $2
            }
            END {
                if (value != "") print value
            }
        ' "$raw")
    fi

    local change
    change=$(awk '
        /^Change / {
            value = $2
        }
        END {
            if (value != "") print value
        }
    ' "$raw")

    local status="ok"

    if [[ "$rc" -ne 0 ]]; then
        status="error"
    fi

    if grep -qi "neighbor list overflow" "$raw"; then
        status="overflow"
    fi

    if [[ -z "$time_s" ]]; then
        status="missing_time"
    fi

    if [[ -z "$change" ]]; then
        change="nan"
    fi

    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
        "$n" "$nsteps" "$force_block" "$use_openmp" "$use_graphs" "$run" \
        "$time_s" "$change" "$status" "$raw" >> "$OUT_RUNS"
}

set -e

for force_block in "${FORCE_BLOCKS[@]}"; do
    for use_openmp in "${OPENMP_MODES[@]}"; do
        for use_graphs in "${CUDA_GRAPH_MODES[@]}"; do
            compile_variant "$force_block" "$use_openmp" "$use_graphs"

            for cfg in "${CONFIGS[@]}"; do
                read -r n nsteps <<< "$cfg"

                for run in $(seq 1 "$RUNS"); do
                    run_one "$n" "$nsteps" "$force_block" "$use_openmp" "$use_graphs" "$run"
                done
            done
        done
    done
done

awk -F, '
BEGIN {
    OFS = ",";
    print "n,nsteps,force_block,openmp,cuda_graphs,runs,avg_time_s,min_time_s,max_time_s,avg_change,status";
}
NR > 1 {
    key = $1 OFS $2 OFS $3 OFS $4 OFS $5;
    seen[key] = 1;

    if ($9 == "ok") {
        runs[key]++;
        sum_time[key] += $7;
        sum_change[key] += $8;

        if (!(key in min_time) || $7 < min_time[key]) {
            min_time[key] = $7;
        }

        if (!(key in max_time) || $7 > max_time[key]) {
            max_time[key] = $7;
        }
    } else {
        bad[key] = 1;
    }
}
END {
    for (key in seen) {
        if (runs[key] > 0) {
            status = bad[key] ? "partial" : "ok";
            print key, runs[key], sum_time[key] / runs[key], min_time[key], max_time[key], sum_change[key] / runs[key], status;
        } else {
            print key, 0, "", "", "", "", "failed";
        }
    }
}
' "$OUT_RUNS" | sort -t, -k1,1n -k2,2n -k3,3n -k4,4n -k5,5n > "$OUT_SUMMARY"

echo "Done."
echo "Run CSV:     $OUT_RUNS"
echo "Summary CSV: $OUT_SUMMARY"