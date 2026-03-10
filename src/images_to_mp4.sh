#!/usr/bin/env bash

# Usage:
# ./images_to_mp4.sh input_folder output.mp4 [fps]

INPUT_DIR="$1"
OUTPUT_MP4="$2"
FPS="5"

if [[ -z "$INPUT_DIR" || -z "$OUTPUT_MP4" ]]; then
    echo "Usage: $0 input_folder output.mp4 [fps]"
    exit 1
fi

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Error: input folder does not exist"
    exit 1
fi

ffmpeg -y \
    -framerate "$FPS" \
    -pattern_type glob \
    -i "$INPUT_DIR/*.jpg" \
    -c:v libx264 \
    -pix_fmt yuv420p \
    -movflags +faststart \
    "$OUTPUT_MP4"
