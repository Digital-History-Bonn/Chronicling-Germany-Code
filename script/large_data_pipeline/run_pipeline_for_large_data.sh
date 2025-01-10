DATA_PATH=$1
FTP_ARGS=$2
IMAGES_PER_ITERATION=$3
AVAILABLE_TIME=$4

MAX_TIME=0
INIT=$(date +%s)

ITERATION_NUMBER=0

while :
do
  ((ITERATION_NUMBER++))
  START=$(date +%s)

  python script/large_data_pipeline/line_selector.py -i "${DATA_PATH}/clean_file_list.txt" -o "${DATA_PATH}/get_images.sh" -f $FTP_ARGS -n $IMAGES_PER_ITERATION

  bash script/large_data_pipeline/organize_data.sh $DATA_PATH

  bash script/pipeline.sh "${DATA_PATH}/images/"

  bash script/large_data_pipeline/organize_results.sh $DATA_PATH

  END=$(date +%s)
  DIFF=$(( END - START ))
  if (( DIFF > MAX_TIME )); then
    MAX_TIME=$DIFF
  fi
  echo "Iteration ${ITERATION_NUMBER} took ${DIFF} seconds."

  DIFF=$(( END - INIT ))
  TIME_SINCE_START=$(( MAX_TIME + DIFF ))
  if (( TIME_SINCE_START * 11 / 10 > AVAILABLE_TIME )); then
    break
  fi
done
END=$(date +%s)
DIFF=$(( END - START ))
echo "Total execution time: ${DIFF} seconds"
echo "Max execution time of an iteration: ${MAX_TIME}"