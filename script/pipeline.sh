#!/bin/bash

# Assign positional parameters to variables
DATA_DIR=$1
ENV_NAME=${2:-'pipeline'}
PROCESS_COUNT=${3:-1}
THREAD_COUNT=${4:-1}
MODEL_LAYOUT="models/layout_2024-05-28.pt"
MODEL_BASELINE="models/baseline_2024-06-01"
MODEL_OCR="models/ocr_2024-09-24.mlmodel"
LAYOUT_PARAMS="-th 0.6 -a dh_segment -s 0.5 -e -bt 200"
PAGE_DIR="${DATA_DIR}/page"

# Activate conda environment and set PYTHONPATH for layout segmentation
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# Run layout segmentation prediction
python -m cgprocess.layout_segmentation.predict -d "$DATA_DIR" -m "$MODEL_LAYOUT" $LAYOUT_PARAMS
# Run baseline detection prediction
python -m cgprocess.baseline_detection.predict -i "$DATA_DIR" -l "$PAGE_DIR" -o "$PAGE_DIR" -m "$MODEL_BASELINE" -t $THREAD_COUNT -p $PROCESS_COUNT
# Run OCR prediction
python -m cgprocess.OCR.LSTM.predict -i "$DATA_DIR" -l "$PAGE_DIR" -o "$PAGE_DIR" -m "$MODEL_OCR" -t $THREAD_COUNT -p $PROCESS_COUNT
