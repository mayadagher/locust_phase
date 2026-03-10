# Function to generate tiles from images.
#!/usr/bin/env bash
set -euo pipefail

tile_one_image() {
    img="$1" # First input is image path
    OUTPUT_DIR="$2" # Second input is output directory

    TILE=320
    STEP=224 # Overlap (0.3) -> 320*(1-0.3) = 224
    RESIZE=640

    base=$(basename "$img" .jpg) # Use image name without extension for beginning of output tile names
    done_file="$DONE_DIR/${base}.done"

    # Skip only if image was fully completed
    if [[ -f "$done_file" ]]; then
        echo "Skipping $base (already complete)"
        return
    fi

    # Get image size (assume square)
    SIZE=$(vipsheader -f width "$img")
    RADIUS=$((SIZE / 2))
    CX=$RADIUS
    CY=$RADIUS

    iy=0
    for ((y=0; y<=SIZE-TILE; y+=STEP)); do
        ix=0
        for ((x=0; x<=SIZE-TILE; x+=STEP)); do

            # Tile center
            cx_tile=$((x + TILE/2))
            cy_tile=$((y + TILE/2))

            dx=$((cx_tile - CX))
            dy=$((cy_tile - CY))

            # Skip tiles outside circular arena
            if (( dx*dx + dy*dy > RADIUS*RADIUS )); then
                ((ix++))
                continue
            fi

            out="${OUTPUT_DIR}/${base}_x${x}_y${y}_ix${ix}_iy${iy}.jpg"

            vips crop "$img" /dev/stdout $x $y $TILE $TILE | \
            vips resize /dev/stdin "$out" $(echo "$RESIZE/$TILE" | bc -l)

            ((ix++))
        done
        ((iy++))
    done


    # Atomically mark completion
    touch "$done_file"
    echo "Completed $base"
}

export -f tile_one_image
