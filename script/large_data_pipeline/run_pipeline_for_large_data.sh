DATA_PATH=$1
FTP_ARGS=$2
IMAGES_PER_ITERATION=$3
AVAILABLE_TIME=$4

MAX_TIME = 0
INIT=$(date +%s)

while :
do
  START=$(date +%s)

  python script/large_data_pipeline/line_selector.py -i "${DATA_PATH}file_list.txt" -o "${DATA_PATH}get_images.sh" -f FTP_ARGS -n IMAGES_PER_ITERATION

  bash script/large_data_pipeline/organize_data.sh DATA_PATH

  bash script/large_data_pipeline/pipeline.sh DATA_PATH

  bash script/large_data_pipeline/organize_results.sh DATA_PATH

  END=$(date +%s)
  DIFF=$(( $END - $START ))
  if [ $DIFF > $MAX_TIME ]; then
    $MAX_TIME = $DIFF
  fi

  if [ ($MAX_TIME + ($END - $INIT)) * 1.1 > AVAILABLE_TIME ]; then
    break
  fi
done