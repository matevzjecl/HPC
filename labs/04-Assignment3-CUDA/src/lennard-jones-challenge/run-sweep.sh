#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --partition=gpu
#SBATCH --job-name=lennard-jones
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=2
#SBATCH --nodes=1
#SBATCH --output=lj_out.log

# LOAD MODULES
module load CUDA

# FOLDERS
mkdir -p results
mkdir -p results/raw

# BUILD
make

# RUN SETTINGS
EXEC="./lj.out"

N_VALUES="1000 2000 4000 8000 16000 20000"
NSTEPS_VALUES="1000 2000 5000 10000"
R_SKIN_VALUES="0.2 0.4 0.6 0.8 1.0"

# Separate CUDA block-size sweeps
FORCE_BLOCK_SIZES="32 64 128 256 512 1024"
KE_BLOCK_SIZES="32 64 128 256 512 1024"
UPDATE_BLOCK_SIZES="32 64 128 256 512 1024"
REBUILD_BLOCK_SIZES="32 64 128 256 512 1024"

RUNS=5
WARMUP_RUNS=5

JOB_ID="${SLURM_JOB_ID:-local}"
OUT="results/lj3d_sweep_${JOB_ID}.csv"
AVG_OUT="results/lj3d_sweep_${JOB_ID}_avg.csv"

# WARMUP RUNS
# These are not included in the CSV results.
WARMUP_N="$(echo "$N_VALUES" | awk '{ print $1 }')"
WARMUP_NSTEPS="$(echo "$NSTEPS_VALUES" | awk '{ print $1 }')"
WARMUP_R_SKIN="$(echo "$R_SKIN_VALUES" | awk '{ print $1 }')"
WARMUP_FORCE_BLOCK="$(echo "$FORCE_BLOCK_SIZES" | awk '{ print $1 }')"
WARMUP_KE_BLOCK="$(echo "$KE_BLOCK_SIZES" | awk '{ print $1 }')"
WARMUP_UPDATE_BLOCK="$(echo "$UPDATE_BLOCK_SIZES" | awk '{ print $1 }')"
WARMUP_REBUILD_BLOCK="$(echo "$REBUILD_BLOCK_SIZES" | awk '{ print $1 }')"

export LJ_R_SKIN="$WARMUP_R_SKIN"
export LJ_FORCE_BLOCK_SIZE="$WARMUP_FORCE_BLOCK"
export LJ_KE_BLOCK_SIZE="$WARMUP_KE_BLOCK"
export LJ_UPDATE_BLOCK_SIZE="$WARMUP_UPDATE_BLOCK"
export LJ_REBUILD_BLOCK_SIZE="$WARMUP_REBUILD_BLOCK"

echo "===== Warmup runs ====="
for warmup in $(seq 1 "$WARMUP_RUNS"); do
    warmup_log="results/raw/warmup_${JOB_ID}_run${warmup}.log"

    echo "Warmup $warmup/$WARMUP_RUNS: n=$WARMUP_N nsteps=$WARMUP_NSTEPS r_skin=$WARMUP_R_SKIN force=$WARMUP_FORCE_BLOCK ke=$WARMUP_KE_BLOCK update=$WARMUP_UPDATE_BLOCK rebuild=$WARMUP_REBUILD_BLOCK"

    srun "$EXEC" "$WARMUP_N" "$WARMUP_NSTEPS" > "$warmup_log" 2>&1
done

echo
echo "===== Starting measured sweep ====="

echo "n,nsteps,r_skin,force_block,ke_block,update_block,rebuild_block,run,time_s,status,start_total,final_total,abs_delta,rel_delta,raw_log" > "$OUT"

for n in $N_VALUES; do
    for nsteps in $NSTEPS_VALUES; do
        for r_skin in $R_SKIN_VALUES; do
            for force_block in $FORCE_BLOCK_SIZES; do
                for ke_block in $KE_BLOCK_SIZES; do
                    for update_block in $UPDATE_BLOCK_SIZES; do
                        for rebuild_block in $REBUILD_BLOCK_SIZES; do
                            for run in $(seq 1 "$RUNS"); do

                                export LJ_R_SKIN="$r_skin"

                                export LJ_FORCE_BLOCK_SIZE="$force_block"
                                export LJ_KE_BLOCK_SIZE="$ke_block"
                                export LJ_UPDATE_BLOCK_SIZE="$update_block"
                                export LJ_REBUILD_BLOCK_SIZE="$rebuild_block"

                                raw_log="results/raw/n${n}_steps${nsteps}_skin${r_skin}_force${force_block}_ke${ke_block}_update${update_block}_rebuild${rebuild_block}_run${run}.log"

                                echo "===== n=$n nsteps=$nsteps r_skin=$r_skin force=$force_block ke=$ke_block update=$update_block rebuild=$rebuild_block run=$run/$RUNS ====="

                                srun "$EXEC" "$n" "$nsteps" > "$raw_log" 2>&1
                                status=$?

                                # Example:
                                # Simulation time 1000 steps: 0.242 seconds
                                time_s="$(
                                    awk '
                                        BEGIN { val = "" }

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
                                    ' "$raw_log"
                                )"

                                # Parse starting total energy if the program prints it.
                                # Supported examples:
                                # Initial E: -3325.3988
                                # Start E:   -3325.3988
                                # start_total -3325.3988
                                start_total="$(
                                    awk '
                                        BEGIN { val = "" }

                                        /start_total/ || /start total/ || /Start total/ || /initial total/ || /Initial total/ || /Initial E:/ || /initial E:/ || /Start E:/ || /start E:/ {
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
                                    ' "$raw_log"
                                )"

                                # Parse final total energy.
                                # Example:
                                # Final E:  -3325.4318 | delta: -0.0330
                                final_total="$(
                                    awk '
                                        BEGIN { val = "" }

                                        /final_total/ || /final total/ || /Final total/ || /Final E:/ || /final E:/ {
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
                                    ' "$raw_log"
                                )"

                                # Parse energy delta from Final E line if available.
                                # Example:
                                # Final E:  -3325.4318 | delta: -0.0330
                                energy_delta="$(
                                    awk '
                                        BEGIN { val = "" }

                                        /Final E:/ || /final E:/ {
                                            for (i = 1; i <= NF; i++) {
                                                if ($i == "delta:" && (i + 1) <= NF) {
                                                    val = $(i + 1)
                                                }
                                            }
                                        }

                                        END {
                                            print val
                                        }
                                    ' "$raw_log"
                                )"

                                # If start_total was not printed, reconstruct it:
                                # delta = final - start  =>  start = final - delta
                                if [[ -z "$start_total" && -n "$final_total" && -n "$energy_delta" ]]; then
                                    start_total="$(
                                        awk -v f="$final_total" -v d="$energy_delta" '
                                            BEGIN {
                                                printf "%.4f", f - d
                                            }
                                        '
                                    )"
                                fi

                                abs_delta=""
                                rel_delta=""

                                if [[ -n "$start_total" && -n "$final_total" ]]; then
                                    read -r abs_delta rel_delta < <(
                                        awk -v s="$start_total" -v f="$final_total" '
                                            BEGIN {
                                                abs = f - s
                                                denom = s

                                                if (denom < 0) {
                                                    denom = -denom
                                                }

                                                if (denom > 1e-12) {
                                                    rel = abs / denom
                                                } else {
                                                    rel = 0.0
                                                }

                                                printf "%.12g %.12g\n", abs, rel
                                            }
                                        '
                                    )
                                fi

                                echo "$n,$nsteps,$r_skin,$force_block,$ke_block,$update_block,$rebuild_block,$run,$time_s,$status,$start_total,$final_total,$abs_delta,$rel_delta,$raw_log" >> "$OUT"

                            done
                        done
                    done
                done
            done
        done
    done
done

echo "n,nsteps,r_skin,force_block,ke_block,update_block,rebuild_block,runs,avg_time_s,min_time_s,max_time_s,avg_abs_delta,avg_rel_delta" > "$AVG_OUT"

awk -F, '
    NR > 1 && $10 == 0 && $9 != "" {
        key = $1 "," $2 "," $3 "," $4 "," $5 "," $6 "," $7

        count[key]++
        time_sum[key] += $9

        if (!(key in min_time) || $9 < min_time[key]) {
            min_time[key] = $9
        }

        if (!(key in max_time) || $9 > max_time[key]) {
            max_time[key] = $9
        }

        if ($13 != "") {
            abs_sum[key] += $13
            abs_count[key]++
        }

        if ($14 != "") {
            rel_sum[key] += $14
            rel_count[key]++
        }
    }

    END {
        for (key in count) {
            avg_time = time_sum[key] / count[key]

            avg_abs = ""
            avg_rel = ""

            if (abs_count[key] > 0) {
                avg_abs = abs_sum[key] / abs_count[key]
            }

            if (rel_count[key] > 0) {
                avg_rel = rel_sum[key] / rel_count[key]
            }

            printf "%s,%d,%.3f,%.3f,%.3f", key, count[key], avg_time, min_time[key], max_time[key]

            if (avg_abs != "") {
                printf ",%.12g", avg_abs
            } else {
                printf ","
            }

            if (avg_rel != "") {
                printf ",%.12g", avg_rel
            } else {
                printf ","
            }

            printf "\n"
        }
    }
' "$OUT" | sort -t, -k1,1n -k2,2n -k3,3n -k4,4n -k5,5n -k6,6n -k7,7n >> "$AVG_OUT"

echo
echo "Done."
echo "Raw results:     $OUT"
echo "Average results: $AVG_OUT"