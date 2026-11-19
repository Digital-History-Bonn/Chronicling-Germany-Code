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

# Stage settings, overridable by environment variables. The defaults are the
# fastest settings measured on Marvin for one A40 with 8 CPU cores
# (jobs 29181911 and 29125662). -p counts processes PER VISIBLE GPU, so on a
# node with more GPUs keep processes x threads x GPUs at about the number of
# CPU cores the job has.
#   BASELINE_ARGS     baseline stage options (default "-t 1 -p 8")
#   BASELINE_THREADS  OMP/MKL/OpenBLAS threads of the baseline stage (default 1; "" = not limited)
#   OCR_ENV           conda env for the OCR stage (default: ENV_NAME); needs Kraken 7.1
#   OCR_ARGS          OCR stage options (default "-t 1 -p 4 -b 32 -w 0"); keep -t 1, Kraken 7's
#                     recognition model is not thread-safe
#   OCR_THREADS       OMP/MKL/OpenBLAS threads of the OCR stage (default 1; "" = not limited)
BASELINE_ARGS=${BASELINE_ARGS:-"-t 1 -p 8"}
BASELINE_THREADS=${BASELINE_THREADS-1}
OCR_ENV=${OCR_ENV:-$ENV_NAME}
OCR_ARGS=${OCR_ARGS:-"-t 1 -p 4 -b 32 -w 0"}
OCR_THREADS=${OCR_THREADS-1}

# Every stage runs this checkout's code, not whatever cgprocess the env has installed.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
stage_time() { echo "[pipeline.sh] $1 $(date +%s.%N)"; }
limit_threads() {  # limit_threads <n>: no-op for ""
    if [[ -n "$1" ]]; then
        export OMP_NUM_THREADS=$1 MKL_NUM_THREADS=$1 OPENBLAS_NUM_THREADS=$1
    fi
}

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# Run layout segmentation prediction
stage_time layout_start
python -m cgprocess.layout_segmentation.predict -d "$DATA_DIR" -m "$MODEL_LAYOUT" $LAYOUT_PARAMS -t 32
# Run baseline detection prediction
stage_time baseline_start
(
    limit_threads "$BASELINE_THREADS"
    python -m cgprocess.baseline_detection.predict -i "$DATA_DIR" -l "$PAGE_DIR" -o "$PAGE_DIR" -m "$MODEL_BASELINE" $BASELINE_ARGS
)
# Run OCR prediction
stage_time ocr_start
(
    conda activate "$OCR_ENV"
    limit_threads "$OCR_THREADS"
    python -m cgprocess.OCR.LSTM.predict -i "$DATA_DIR" -l "$PAGE_DIR" -o "$PAGE_DIR" -m "$MODEL_OCR" $OCR_ARGS
)
stage_time ocr_end
