#!/bin/bash
#SBATCH --reservation=fri
#SBATCH --job-name=seam_bench_sweep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --output=slurm/%x_%j.log

set -euo pipefail

# =========================
# Manual parameters
# =========================
IMG="test_images/1024x768.png"
REMOVE_PIXELS=128
PAR=1
USE_TRI=1
RUNS=10
TRI_VALUES="24 36 40 48 56 64 72 88 96"

export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load numactl

mkdir -p bin logs summaries slurm output_images

gcc -O3 seam_carving.c -lm -lnuma --openmp -o bin/seam_carving

img_name=$(basename "$IMG")
img_base="${img_name%.png}"
cpus="$SLURM_CPUS_PER_TASK"

for TRI_N in $TRI_VALUES; do
  summary_file="summaries/${img_base}_par${PAR}_tri${USE_TRI}_cpus${cpus}_tn${TRI_N}_avg.txt"
  out_img="output_images/${img_base}_par${PAR}_tri${USE_TRI}_cpus${cpus}_tn${TRI_N}-out.png"

  sum_s=0

  for run in $(seq 1 "$RUNS"); do
    run_log="logs/${img_base}_par${PAR}_tri${USE_TRI}_cpus${cpus}_tn${TRI_N}_run${run}.log"

    {
      echo "image=$IMG"
      echo "remove_pixels=$REMOVE_PIXELS"
      echo "parallel=$PAR"
      echo "use_triangles=$USE_TRI"
      echo "cpus=$cpus"
      echo "triangle_count=$TRI_N"
      echo "run=$run"
      echo "output_image=$out_img"
    } > "$run_log"

    srun bin/seam_carving "$IMG" "$out_img" "$REMOVE_PIXELS" "$PAR" "$USE_TRI" "$TRI_N" \
      >> "$run_log" 2>&1

    runtime_s=$(grep 'runtime_seconds=' "$run_log" | tail -n 1 | cut -d= -f2)

    if [ -z "$runtime_s" ]; then
      echo "Failed to find runtime_seconds in $run_log"
      exit 1
    fi

    echo "measured_runtime_seconds=$runtime_s" >> "$run_log"

    sum_s=$(awk "BEGIN {printf \"%.6f\", $sum_s + $runtime_s}")
  done

  avg_s=$(awk "BEGIN {printf \"%.6f\", $sum_s / $RUNS}")
  avg_ms=$(awk "BEGIN {printf \"%.2f\", $avg_s * 1000}")

  {
    echo "image=$IMG"
    echo "remove_pixels=$REMOVE_PIXELS"
    echo "parallel=$PAR"
    echo "use_triangles=$USE_TRI"
    echo "cpus=$cpus"
    echo "triangle_count=$TRI_N"
    echo "runs=$RUNS"
    echo "output_image=$out_img"
    echo "avg_runtime_seconds=$avg_s"
    echo "avg_ms=$avg_ms"
  } > "$summary_file"

  echo "Finished TRI_N=$TRI_N avg_ms=$avg_ms"
done