#!/bin/bash
set -euo pipefail

SUBMIT_DIR="$PWD"
STAMP=$(date +%Y%m%d_%H%M%S_%N)
SNAPDIR="$SUBMIT_DIR/job_snapshots/lenia_cuda_$STAMP"

mkdir -p "$SNAPDIR"
mkdir -p "$SUBMIT_DIR/logs"

cp run_cuda_sweep.sh "$SNAPDIR"/
cp Makefile "$SNAPDIR"/
cp -r src "$SNAPDIR"/

sbatch \
  --chdir="$SNAPDIR" \
  --output="$SUBMIT_DIR/logs/%x_%j.log" \
  "$SNAPDIR/run_cuda_sweep.sh"