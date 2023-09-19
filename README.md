# Newspaper Image Segmentation

An image segmentation model for the detection of structure in historical newspaper pages

Model based on: https://arxiv.org/abs/1804.10371

Tested on Python 3.10.4. Other requirements in requirements.txt

## Code Style
Pylint can be used with pycharm by installing the pylint plugin.

My py can be used with pycharm by installing the my py plugin.

## TensorBoard
logs can be accessed trough command line
````shell
tensorboard --logdir logs/runs
````

or with magic commands in jupiter notebooks
````
%load_ext tensorboard
%tensorboard --logdir logs/runs
````

## Preprocessing and Training

Before starting the training process all data has to be converted. 
This command loads xml annotation data and converts it to .npy files.
````
python script/convert_xml.py -a annotations/ -o targets/
````

The Training script assumes, that the supplied data folder contains 'targets' and 'images' folders.
````
python src/news_seg/train.py -e 100 -n experiment_name -b 64 -d data_folder/  -g 4 -w 32
````

If the training process has to be intrrupted, training can be continued by executing this command.
````
python src/news_seg/train.py -e 100 -n experiment_name -b 64 -d data_folder/  -l model_name -ls -g 4 -w 32
````

## Slurm

In the slurm/ folder are examples of the used slurm files.
