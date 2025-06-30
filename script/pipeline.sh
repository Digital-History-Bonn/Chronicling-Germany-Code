#!/bin/bash

# Assign positional parameters to variables
DATA_DIR=$1
ENV_NAME=${2:-'pipeline'}
PROCESS_COUNT=${3:-1}
THREAD_COUNT=${4:-1}
MODEL_LAYOUT="models/Chronicling-Germany-Dataset-main-models/models/layout_2025-05-14.pt"
MODEL_BASELINE="models/Chronicling-Germany-Dataset-main-models/models/baseline_2025-05-19.pt"
MODEL_OCR="models/Chronicling-Germany-Dataset-main-models/models/ocr_2025-05-14.mlmodel"
LAYOUT_PARAMS="-th 0.6 -a dh_segment -s 0.5 -e -bt 100"
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
