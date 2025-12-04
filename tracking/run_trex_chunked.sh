
# File for splitting analysis into chunks for TRex\

CONFIG_FILE='/Users/mayadagher/Documents/Mutual_inf/tracking/locust_data/trex_inputs/20230329.settings'
PV_FILE='/Users/mayadagher/Documents/Mutual_inf/tracking/locust_data/trex_inputs/20230329.pv'
VIDEO_FILE='/Users/mayadagher/Documents/Mutual_inf/tracking/locust_data/trex_inputs/20230329.mp4'
OUTPUT_DIR='/Users/mayadagher/Documents/Mutual_inf/tracking/locust_data/trex_outputs'
CHUNK=6000
OVERLAP=100

# --- get total frames (using ffprobe) ---
TOTAL_FRAMES=$(ffprobe -v error -select_streams v:0 \
  -show_entries stream=nb_frames \
  -of default=nokey=1:noprint_wrappers=1 "$VIDEO_FILE")

echo "Total frames: $TOTAL_FRAMES"

# --- iterate over chunks ---
START=0
INDEX=0

while [ "$START" -lt "$TOTAL_FRAMES" ]; do

 trap 'echo "Interrupted by user. Exiting..."; exit 130' INT


  END=$((START + CHUNK))
  if [ "$END" -gt "$TOTAL_FRAMES" ]; then
    END=$TOTAL_FRAMES
  fi
  echo "Processing chunk $INDEX: frames $START to $END"

 # CHUNK_DIR="$\{OUTPUT_DIR}/chunk_$\{INDEX}"
 # mkdir -p "$CHUNK_DIR"

 # Run TRex tracking for this chunk\
 echo "Running new trex command"
 trex -i "$PV_FILE" -task track -settings_file "$CONFIG_FILE" -analysis_range "[${START},${END}]" -p "batch_${INDEX}" -auto_quit|| {
  echo "TRex failed on chunk $INDEX. Stopping further processing."
  break
 }

  # Advance to next window (overlap = 100), if not at end
  if ["$END" -eq "$TOTAL_FRAMES"]; then
  	break
  fi
  START=$((END - OVERLAP))
  INDEX=$((INDEX + 1))
done

echo "All chunks processed and saved to $OUTPUT_DIR"
