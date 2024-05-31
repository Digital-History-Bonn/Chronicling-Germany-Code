eval "$(conda shell.bash hook)"

conda activate layout
export PYTHONPATH=${PYTHONPATH}:

python src/news_seg/predict.py -d data/PipelineTest/test -m models/model_neurips_dh_segment_3.0_best.pt -p 5760 7680 -t 0.6 -a dh_segment -s 0.5 -e -bt 200
python src/baseline_detection/pero/predict.py -i data/PipelineTest/test -l data/PipelineTest/test/page -o data/PipelineTest/test/page -m models/baselineFinal2_baseline_aug_e200_es

conda activate ocr
export PYTHONPATH=${PYTHONPATH}:
python src/OCR/predict.py -i data/PipelineTest/test -l data/PipelineTest/test/page -o data/PipelineTest/test/page -m models/ocr_scratch_best.mlmodel

