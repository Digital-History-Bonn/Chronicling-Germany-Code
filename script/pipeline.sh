#!/bin/bash

# Assign positional parameters to variables
DATA_DIR=$1
ENV_NAME=${2:-'pipeline'}
PROCESS_COUNT=${3:-1}
THREAD_COUNT=${4:-1}
MODEL_LAYOUT="../models/Chronicling-Germany-Dataset-main-models/models/layout_2025-05-14.pt"
MODEL_BASELINE="../models/Chronicling-Germany-Dataset-main-models/models/baseline_2025-05-19.pt"
MODEL_OCR="../models/Chronicling-Germany-Dataset-main-models/models/ocr_2025-05-14.mlmodel"
LAYOUT_PARAMS="-th 0.6 -a dh_segment -s 0.25 -e -bt 100"
PAGE_DIR="${DATA_DIR}/page"
# Optional OCR-stage settings (environment variables; unset = the settings above):
#   OCR_ENV      conda env for the OCR stage only, e.g. newspapers_k7 (Kraken 7.1);
#                this repo's src/ is then put on PYTHONPATH, so the env needs no cgprocess install
#   OCR_ARGS     options for the OCR stage instead of "-t 4 -p 4" (-p counts processes per GPU)
#   OCR_THREADS  if set, limits OMP/MKL/OpenBLAS threads of the OCR stage to this number
OCR_ARGS=${OCR_ARGS:-"-t 4 -p 4"}
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
stage_time() { echo "[pipeline.sh] $1 $(date +%s.%N)"; }

# Activate conda environment and set PYTHONPATH for layout segmentation
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# Run layout segmentation prediction
stage_time layout_start
python -m cgprocess.layout_segmentation.predict -d "$DATA_DIR" -m "$MODEL_LAYOUT" $LAYOUT_PARAMS -t 32
# Run baseline detection prediction
stage_time baseline_start
python -m cgprocess.baseline_detection.predict -i "$DATA_DIR" -l "$PAGE_DIR" -o "$PAGE_DIR" -m "$MODEL_BASELINE" -t 4 -p 1
# Run OCR prediction
stage_time ocr_start
(
    if [[ -n "${OCR_ENV:-}" ]]; then
        conda activate "$OCR_ENV"
        export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
    fi
    if [[ -n "${OCR_THREADS:-}" ]]; then
        export OMP_NUM_THREADS=$OCR_THREADS MKL_NUM_THREADS=$OCR_THREADS OPENBLAS_NUM_THREADS=$OCR_THREADS
    fi
    python -m cgprocess.OCR.LSTM.predict -i "$DATA_DIR" -l "$PAGE_DIR" -o "$PAGE_DIR" -m "$MODEL_OCR" $OCR_ARGS
)
stage_time ocr_end
