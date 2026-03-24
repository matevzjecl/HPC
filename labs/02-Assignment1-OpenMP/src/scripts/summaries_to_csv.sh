#!/bin/bash
set -euo pipefail

# Usage:
#   ./summaries_to_csv.sh summaries output.csv
#
# Example:
#   ./summaries_to_csv.sh summaries triangle_results.csv

INPUT_DIR="${1:-summaries}"
OUTPUT_CSV="${2:-summary_results.csv}"

if [ ! -d "$INPUT_DIR" ]; then
  echo "Error: directory '$INPUT_DIR' does not exist."
  exit 1
fi

# Header
echo "image_resolution,remove_pixels,parallel,use_triangles,cpus,triangle_width,runs,avg_runtime_seconds,avg_runtime_ms" > "$OUTPUT_CSV"

found_any=0

for file in "$INPUT_DIR"/*_avg.txt; do
  [ -e "$file" ] || continue
  found_any=1

  image_resolution=""
  remove_pixels=""
  parallel=""
  use_triangles=""
  cpus=""
  triangle_width=""
  runs=""
  avg_runtime_seconds=""
  avg_runtime_ms=""

  while IFS='=' read -r key value; do
    case "$key" in
      image)
        img_name=$(basename "$value")
        image_resolution="${img_name%.png}"
        ;;
      remove_pixels) remove_pixels="$value" ;;
      parallel) parallel="$value" ;;
      use_triangles) use_triangles="$value" ;;
      cpus) cpus="$value" ;;
      triangle_count) triangle_width="$value" ;;
      runs) runs="$value" ;;
      avg_runtime_seconds) avg_runtime_seconds="$value" ;;
      avg_ms) avg_runtime_ms="$value" ;;
    esac
  done < "$file"

  echo "${image_resolution},${remove_pixels},${parallel},${use_triangles},${cpus},${triangle_width},${runs},${avg_runtime_seconds},${avg_runtime_ms}" >> "$OUTPUT_CSV"
done

if [ "$found_any" -eq 0 ]; then
  echo "No *_avg.txt files found in '$INPUT_DIR'."
  rm -f "$OUTPUT_CSV"
  exit 1
fi

echo "Created CSV: $OUTPUT_CSV"