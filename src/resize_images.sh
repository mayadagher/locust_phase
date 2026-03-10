#!/usr/bin/env bash

# Usage:
# ./resize_jpegs.sh input_dir output_dir

set -euo pipefail

INPUT_DIR="$1"
OUTPUT_DIR="$2"

mkdir -p "$OUTPUT_DIR"

for img in "$INPUT_DIR"/*.jpg "$INPUT_DIR"/*.jpeg; do
    [ -e "$img" ] || continue  # skip if no matches

    filename=$(basename "$img")

    magick "$img" \
        -resize 1920x1920! \
        "$OUTPUT_DIR/$filename"
done