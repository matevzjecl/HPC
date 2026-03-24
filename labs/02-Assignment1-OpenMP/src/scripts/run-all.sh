#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=seam_carving
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=%x_%j.log
#SBATCH --hint=nomultithread

set -euo pipefail

export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load numactl

timestamp=$(date +"%Y%m%d-%H%M%S")
run_log="seam_carving_runs_${SLURM_JOB_ID}_${timestamp}.log"

log() {
    echo "[$(date +"%Y%m%d-%H:%M:%S")] $*" | tee -a "$run_log"
}

log "Job started"
log "Job ID: ${SLURM_JOB_ID:-N/A}"
log "CPUs per task: ${SLURM_CPUS_PER_TASK:-N/A}"
log "OMP_NUM_THREADS=$OMP_NUM_THREADS"

log "Compiling seam_carving.c"
gcc -O3 seam_carving.c -o seam_carving -lm -lnuma -fopenmp
log "Compilation finished"

images=(
    "720x480"
    "1024x768"
    "1920x1200"
    "3840x2160"
    "7680x4320"
)

pixels_to_remove=128

for img in "${images[@]}"; do
    input="test_images/${img}.png"

    # Sequential
    output_seq="test_images/${img}-out-seq.png"
    log "START mode=sequential image=${img} input=${input} output=${output_seq} remove=${pixels_to_remove}"
    /usr/bin/time -f "[%x] elapsed=%E user=%U sys=%S maxrss=%MKB" \
        srun ./seam_carving "$input" "$output_seq" "$pixels_to_remove" 0 \
        2>&1 | tee -a "$run_log"
    log "END   mode=sequential image=${img}"

    # Parallel
    output_par="test_images/${img}-out.png"
    log "START mode=parallel   image=${img} input=${input} output=${output_par} remove=${pixels_to_remove}"
    /usr/bin/time -f "[%x] elapsed=%E user=%U sys=%S maxrss=%MKB" \
        srun ./seam_carving "$input" "$output_par" "$pixels_to_remove" 1 \
        2>&1 | tee -a "$run_log"
    log "END   mode=parallel   image=${img}"
done

log "Job finished"
log "Detailed run log saved to $run_log"