#!/bin/bash

# Assign positional parameters to variables
DATA_DIR=$1
PROCESS_COUNT={$2:-1}
THREAD_COUNT={$3:-1}
ENV_NAME={$4:-'pipeline'}
MODEL_LAYOUT="models/layout_2024-05-28.pt"
MODEL_BASELINE="models/baseline_2024-06-01"
MODEL_OCR="models/ocr_2024-09-24.mlmodel"
LAYOUT_PARAMS="-p 5760 7680 -t 0.6 -a dh_segment -s 0.5 -e -bt 200"
PAGE_DIR="${DATA_DIR}/page"

# Activate conda environment and set PYTHONPATH for layout segmentation
eval "$(conda shell.bash hook)"
conda activate pipeline
export PYTHONPATH=${PYTHONPATH}:

# Run layout segmentation prediction
python src/layout_segmentation/predict.py -d "$DATA_DIR" -m "$MODEL_LAYOUT" $LAYOUT_PARAMS
# Run baseline detection prediction
python src/baseline_detection/predict.py -i "$DATA_DIR" -l "$PAGE_DIR" -o "$PAGE_DIR" -m "$MODEL_BASELINE" -p $PROCESS_COUNT
# Run OCR prediction
python src/OCR/LSTM/predict.py -i "$DATA_DIR" -l "$PAGE_DIR" -o "$PAGE_DIR" -m "$MODEL_OCR" -t $THREAD_COUNT -p $PROCESS_COUNT
