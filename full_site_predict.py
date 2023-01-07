"""
Module to predict full newspaper pages
"""

import numpy as np
import torch    # type: ignore
from skimage.io import imread, imsave    # type: ignore
from skimage.color import gray2rgb  # type: ignore

from model import DhSegment

IN_CHANNELS, OUT_CHANNELS = 3, 10
MODEL = 'Models/model_PaperParams3.pt'
PAGE = '9431446.jpeg'


def predict(path):
    """
    predict a Page from given Path with given Model
    :param path: path to image of page
    :return: None
    """
    # create model
    model = DhSegment([3, 4, 6, 4], in_channels=IN_CHANNELS, out_channel=OUT_CHANNELS)
    model.float()

    # load model if argument is None it does nothing
    model.load(MODEL)

    # set mean and std in model for normalization
    model.means = torch.tensor((0.485, 0.456, 0.406))
    model.stds = torch.tensor((0.229, 0.224, 0.225))

    model.eval()

    page = imread(path)

    if page.ndim < 3:
        page = gray2rgb(page)

    p_width, p_height = 32 - (page.shape[0] % 32), 32 - (page.shape[1] % 32)
    page = np.pad(page, ((0, p_width), (0, p_height), (0, 0)), 'constant', constant_values=0)

    page = torch.tensor([page])
    page = torch.permute(page, (0, 3, 1, 2))

    return model.predict(page.float()/255)


def save(prediction: torch.tensor) -> None:
    """
    saves prediction in matplotlib image
    :param prediction:
    :return:
    """
    imsave('result_full_site.png', prediction)


def main():
    """
    predicts the page and saves result in image
    :return: None
    """
    pred = predict(PAGE)
    save(pred)


if __name__ == '__main__':
    main()
