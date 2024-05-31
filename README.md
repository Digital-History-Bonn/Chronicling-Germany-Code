# Newspaper Image Segmentation

Our layout recognition model is divided into three subtasks:
1. Layout segmentation
2. Baseline detection
3. Optical character recognition

All tasks are trained and evaluated individually, but work together to extract text from a given input document (see Prediction Pipeline).

## Requirements
Tested on Python 3.10.4.
Two separate environments are needed for layout segmentation and OCR. Layout segmentation and baseline detection use the requirements.txt in the main folder. The OCR uses src/OCR/requirements.txt.

## Prediction Pipeline
To run the complete prediction from an input image to an annotation xml with layout and text our pipeline in predict.sh can be used.
This pipeline needs two conda environments named 'layout' and 'ocr' (see requirements). 


## Layout Segmentation

An image segmentation model for the detection of structure in historical newspaper pages

Model based on: https://arxiv.org/abs/1804.10371

### Preprocessing and Training
Depending on your setup, you may require `PYTHONPATH=.` in front of the commands below.

Before starting the training process all data has to be converted.
This command loads xml annotation data and converts it to .npy files.
````
python src/news_seg/convert_xml.py -a annotations/ -o targets/
````

The Training script assumes, that the supplied data folder contains 'targets' and 'images' folders.
````
python src/news_seg/train.py -e 100 -n experiment_name -b 64 -d data_folder/  -g 4 -w 32
````

If the training process has to be intrrupted, training can be continued by executing this command.
````
python src/news_seg/train.py -e 100 -n experiment_name -b 64 -d data_folder/  -l model_name -ls -g 4 -w 32
````

### Prediction

Prediction takes in images and processes them with a specified model. The image is processed in its entirety. 
This can lead to cuda out of memory errors, if the resulution is too high.
Furthermore, one has to specifiy an image size, to which the image will be padded. 
If the number of pixel on one of the side is not divisible at least 6 times by 2, the prediction will fail.

If an output folder is specified, images of the prediction will be saved in that folder. However, this option seriously
increases execution time and should only be used for debugging. If the -e option is active, the xml files will be 
exported to a page folder within the data folder. If there are already xml files, those will be overwritten.

Example for calling the predict script.
````
python src/news_seg/predict.py -d ../../data/ -m models/model_best.pt -p 5760 7680 -t 0.6 -s 0.5 -e -bt 100````
````

### Debug
To analyse the models predictions we added a --debug option to the prediction function. With this option the predict
function does not output the prediction, instead it outputs the areas of uncertainty of the models. This areas are all 
pixels that have a predicted probability under the given threshold for the ground truth class.
````
python src/news_seg/predict.py -d data_folder/ -o output_folder/ -m path/to/model/ -a dh_segment -p 5760 7360 -s 0.5 --transkribus-export --debug
````

## Baseline detection

Baseline detection is based on the idea of pero: 
O Kodym, M Hradiš: Page Layout Analysis System for Unconstrained Historic Documents. ICDAR, 2021.

### Preprocessing and Training
The trainings script need targets saved as .npz files. The target can be created by with our preprocessing script.
````
python -m src.baseline_detection.pero.preprocess -i path/to/images -a path/to/annotations -o path/to/output/folder
````
The preprocessed data can then be splited into our train, valid and test split with:
````
python -m src.baseline_detection.split -i path/to/images -a path/to/targets -o path/to/output/folder
````
The training script can then be started with:
````
python -m src.baseline_detection.pero.trainer -n NameOfTheModel -t path/to/train/data -v path/to/train/data -e 200
````

### Prediction
The baseline prediction uses the layout (prediction) to differentiate between different text regions and exclude Table regions.
It can be started with:
````
python -m src.baseline_detection.pero.predict -i path/to/images -l path/to/layout/annotations -o path/to/output/folder -m path/to/model
````
The image folder and the layout folder can be the same, but the name of the image file and the .xml file with the layout annotations must match.

## Optical character recognition
The OCR is based on Kraken (https://kraken.re/main/index.html).

**_NOTE:_**  Kraken uses an older shapely version and is therefore not compatible with our code for layout segmentation and baseline detection. Therefore a separate python enviroment is required for the OCR subtask. The necessary requirements can be found in the reqierements.txt in src/OCR

### Preprocessing and Training
Kraken uses the filename in the .xml file to find the image while training. So the image files should always be in the same folder as the annotation files.
For preprocessing we padded all images and the annotations by 10 pixels. This can be done by:
````
python -m src.OCR.preprocess -i path/to/data  -o path/to/output/folder
````

After that the training can be stared:
````
python -m src.OCR.train -n NameOfTheModel -t path/to/train/data -v path/to/valid/data
````

### Prediction
To predict the Text in an image our tool needs baseline (predictions). The process can be started with:
````
python -m src.OCR.predict -i path/to/images -l path/to/annotations -o path/to/output/folder -m path/to/model
````
Again the image folder and the layout annotation folder can be the same, but the name of the image file and the .xml file with the layout annotations must match.


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

## File names
File names of xml and jpg are identical. Pages are grouped into different time periods specified with a 
specific date, a timeframe of a few months or an entire year. The last 4 characters of each file names are a 
number, that is in the order of publication regarding the specified time period. The number are NOT unique 
over the entire dataset. Most of the pages are from 1866 and contain "1866-06_1866-09" in their file names. 
A few of them are specified to be advertisements with "1866-06_1866-09_Anzeigen". Most of the other pages are 
from 1924, containing "1924" in the file name. Finally, there are a few pages, that have a specific date in their 
file name like "1865-04-27" these are special editions.

## Slurm

In the slurm/ folder are examples of the used slurm files.