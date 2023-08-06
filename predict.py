"""Module for predicting newspaper images with trained models. """

import argparse
import os

import matplotlib.patches as mpatches  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import torch
from PIL import Image  # type: ignore
from numpy import ndarray
from skimage.color import label2rgb  # type: ignore
from torchvision import transforms  # type: ignore
from tqdm import tqdm  # type: ignore

import train

DATA_PATH = "data/newspaper/input/"
RESULT_PATH = "data/output/"

cmap = [(1.0, 0.0, 0.16), (1.0, 0.43843843843843844, 0.0), (0, 0.222, 0.222), (0.36036036036036045, 0.5, 0.5),
        (0.0, 1.0, 0.2389486260454002), (0.8363201911589008, 1.0, 0.0), (0.0, 0.5615942028985507, 1.0),
        (0.0422705314009658, 0.0, 1.0), (0.6461352657004831, 0.0, 1.0), (1.0, 0.0, 0.75)]


def draw_prediction(img: ndarray, path: str):
    """
    Draw prediction with legend. And save it.
    :param img: prediction ndarray
    :param path: path for the prediction to be saved.
    """
    img[-1][-1] = 1
    unique, counts = np.unique(img, return_counts=True)
    print(dict(zip(unique, counts)))
    values = ["UnknownRegion", "caption", "table", "article", "heading", "header", "separator (vertical)",
              "separator (short)", "separator (horizontal)"]
    plt.imshow(label2rgb(img, bg_label=0, colors=cmap))
    plt.axis('off')
    # create a patch (proxy artist) for every color
    patches = [mpatches.Patch(color=cmap[i], label=f"{values[i]}") for i in range(9)]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, loc=4)
    plt.autoscale(tight=True)
    plt.savefig(path, bbox_inches=0, pad_inches=0)
    plt.show()


def get_args() -> argparse.Namespace:
    """defines arguments"""
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--data-path', '-p', type=str,
                        default=DATA_PATH,
                        help='path for folder with images to be segmented')
    parser.add_argument('--result-path', '-r', type=str,
                        default=RESULT_PATH,
                        help='path for folder where prediction images are to be saved')
    parser.add_argument('--model-path', '-m', type=str,
                        default="model.pt",
                        help='path to model .pt file')
    parser.add_argument('--cuda-device', '-c', type=str, default="cuda",
                        help='Cuda device string')
    return parser.parse_args()


def load_image(file: str) -> torch.Tensor:
    """
    Loads image and applies necessary transformation for prdiction.
    :param file: path to image
    :return: Tensor of dimensions (BxCxHxW). In this case, the number of batches will always be 0.
    """
    image = Image.open(args.data_path + file).convert('RGB')
    transform = transforms.PILToTensor()
    data: torch.Tensor = transform(image).float() / 255
    data = torch.unsqueeze(data, dim=0)
    return data


def predict():
    """
    Loads all images from data folder and predicts segmentation.
    """
    device = args.cuda_device if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    file_names = os.listdir(args.data_path)
    model = train.init_model(args.model_path)
    model.eval()
    model.to(device)
    for file in tqdm(file_names, desc='predicting images', total=len(file_names), unit='files'):
        image = load_image(file)

        pred = np.squeeze(model(image.to(device)).detach().cpu().numpy())
        draw_prediction(np.argmax(pred, axis=0), args.result_path + os.path.splitext(file)[0] + '.png')


if __name__ == '__main__':
    args = get_args()

    predict()
