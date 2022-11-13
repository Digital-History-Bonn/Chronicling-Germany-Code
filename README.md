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
