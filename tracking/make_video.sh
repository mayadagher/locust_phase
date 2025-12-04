#!/bin/bash
# make_video.sh
# Usage: ./make_video.sh /path/to/images output.mp4

set -euo pipefail

IMG_DIR="$1"
OUTPUT="$2"
TMP_DIR=$(mktemp -d)
CHUNK_SIZE=20    # how many frames per batch
FPS=5            # output frame rate
CRF=28           # compression quality (lower = better quality, bigger file)
PRESET=veryfast  # ffmpeg encode speed vs file size

# Collect files safely into an array, sorted by filename (chronological order)
IMAGES=()
while IFS= read -r -d '' file; do
    IMAGES+=("$file")
done < <(find "$IMG_DIR" -maxdepth 1 -type f -iname '*.jpg' -print0 | sort -z)

TOTAL=${#IMAGES[@]}
CHUNKS=$(( (TOTAL + CHUNK_SIZE - 1) / CHUNK_SIZE ))

echo "Found $TOTAL images, processing in $CHUNKS chunks..."

CHUNK_VIDS=()

for ((i=0; i<CHUNKS; i++)); do
    START=$((i * CHUNK_SIZE))
    END=$((START + CHUNK_SIZE))
    [ $END -gt $TOTAL ] && END=$TOTAL

    echo "Processing frames $START to $((END-1))..."

    LIST="$TMP_DIR/list_$i.txt"
    : > "$LIST"
    for ((j=START; j<END; j++)); do
        printf "file '%s'\n" "${IMAGES[$j]}" >> "$LIST"
    done

    CHUNK_VIDEO="$TMP_DIR/chunk_$i.mp4"
    ffmpeg -y -r $FPS -f concat -safe 0 -i "$LIST" \
    -vf scale=1920:-1 \
    -c:v libx264 -preset $PRESET -crf $CRF "$CHUNK_VIDEO"

    CHUNK_VIDS+=("$CHUNK_VIDEO")
done

# Concatenate all chunk videos
echo "Concatenating chunks into final video..."
FINAL_LIST="$TMP_DIR/final_list.txt"
: > "$FINAL_LIST"
for vid in "${CHUNK_VIDS[@]}"; do
    printf "file '%s'\n" "$vid" >> "$FINAL_LIST"
done

ffmpeg -y -f concat -safe 0 -i "$FINAL_LIST" -c copy "$OUTPUT"

echo "✅ Video written to $OUTPUT"
rm -rf "$TMP_DIR"
