#!/bin/bash
# Run the whole cgprocess pipeline (layout segmentation, baseline detection, OCR)
# on a folder of page images.
#
# Usage: script/pipeline.sh DATA_DIR [ENV_NAME] [PROCESS_COUNT] [THREAD_COUNT]
#   DATA_DIR       folder with the page images; results go to DATA_DIR/page
#   ENV_NAME       conda env with cgprocess (default "pipeline")
#   PROCESS_COUNT  processes PER VISIBLE GPU for baseline detection and OCR (default 1)
#   THREAD_COUNT   threads per process for baseline detection (default 1)
#
# Stage settings can be overridden with environment variables, or with a config
# file named by PIPELINE_CONFIG that sets them (see script/configs/, e.g.
# PIPELINE_CONFIG=script/configs/marvin_a40.env for one A40 GPU with 8 CPU cores):
#   MODEL_DIR         folder with the layout, baseline and OCR models
#   LAYOUT_PARAMS     layout stage options
#   BASELINE_ARGS     baseline stage options (default "-t THREAD_COUNT -p PROCESS_COUNT")
#   BASELINE_THREADS  OMP/MKL/OpenBLAS threads of the baseline stage (default 1; "" = not limited)
#   OCR_ENV           conda env for the OCR stage (default: ENV_NAME); needs Kraken 7.1
#   OCR_ARGS          OCR stage options (default "-t 1 -p PROCESS_COUNT -b 32 -w 0"); keep -t 1,
#                     Kraken 7's recognition model is not thread-safe
#   OCR_THREADS       OMP/MKL/OpenBLAS threads of the OCR stage (default 1; "" = not limited)
# -p counts processes per visible GPU, so keep processes x threads x GPUs at about the
# number of CPU cores. Limiting the BLAS threads to 1 made the OCR stage 3.4x faster.

DATA_DIR=$1
ENV_NAME=${2:-'pipeline'}
PROCESS_COUNT=${3:-1}
THREAD_COUNT=${4:-1}
PAGE_DIR="${DATA_DIR}/page"

if [[ -n "$PIPELINE_CONFIG" ]]; then
    # shellcheck source=/dev/null
    source "$PIPELINE_CONFIG" || exit 1
fi

MODEL_DIR=${MODEL_DIR:-"models/Chronicling-Germany-Dataset-main-models/models"}
MODEL_LAYOUT="${MODEL_DIR}/layout_2025-05-14.pt"
MODEL_BASELINE="${MODEL_DIR}/baseline_2025-05-19.pt"
MODEL_OCR="${MODEL_DIR}/ocr_2025-05-14.mlmodel"
LAYOUT_PARAMS=${LAYOUT_PARAMS:-"-th 0.6 -a dh_segment -s 0.5 -e -bt 100"}
BASELINE_ARGS=${BASELINE_ARGS:-"-t $THREAD_COUNT -p $PROCESS_COUNT"}
BASELINE_THREADS=${BASELINE_THREADS-1}
OCR_ENV=${OCR_ENV:-$ENV_NAME}
OCR_ARGS=${OCR_ARGS:-"-t 1 -p $PROCESS_COUNT -b 32 -w 0"}
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
conda activate "$ENV_NAME"

# Run layout segmentation prediction
stage_time layout_start
python -m cgprocess.layout_segmentation.predict -d "$DATA_DIR" -m "$MODEL_LAYOUT" $LAYOUT_PARAMS
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
