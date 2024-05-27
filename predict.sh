eval "$(conda shell.bash hook)"

conda activate layout
export PYTHONPATH=${PYTHONPATH}:

python src/news_seg/predict.py -d data/ -m models/model.pt -p 5760 7680 -t 0.6 -a dh_segment -s 0.5 -e -bt 200
python src/baseline_detection/predict.py -i data -l data/page -o data/page

# conda activate ocr ?
python src/ocr/predict.py -i data -l data/page -o data/page

