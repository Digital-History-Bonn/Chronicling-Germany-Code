"""Module for loading Model and predicting images"""
import argparse
import os
from typing import Union, List, Any

import numpy as np
from PIL import Image  # type: ignore
import torch  # type: ignore

from model import DhSegment

IN_CHANNELS, OUT_CHANNELS = 3, 10


def predict(model: DhSegment, file: str) -> np.ndarray:
    """
    predicts an image given as path to file
    :param model: DhSegment
    :param file: path to image
    :return: prediction as ndarray
    """
    img = _get_file(file)

    prediction = model(img.to(DEVICE)).argmax(dim=1).float().cpu()

    return np.array(torch.squeeze(prediction / OUT_CHANNELS)) * 255


def _get_model() -> DhSegment:
    # create model
    model: DhSegment = DhSegment([3, 4, 6, 4], in_channels=IN_CHANNELS, out_channel=OUT_CHANNELS)
    model = model.float()
    model.load(args.model)
    model.to(DEVICE)
    model.eval()
    return model


def _get_file(file: str) -> torch.tensor:
    img = Image.open(file).convert('RGB')
    shape = int(img.size[0] * scale), int(img.size[1] * scale)
    img = img.resize(shape, resample=Image.BICUBIC)

    w_pad, h_pad = (32 - (shape[0] % 32)), (32 - (shape[1] % 32))
    img = np.pad(np.array(img), ((0, h_pad), (0, w_pad), (0, 0)), 'constant', constant_values=0)
    img = np.transpose(torch.tensor(img), (2, 0, 1))
    return torch.unsqueeze(img, dim=0)


def get_output_filenames(output_path: Union[str, List[Any]], input_path: Union[str, List[Any]]) -> \
        Union[str, List[Any]]:
    """returns generated output name or output name from args"""

    return output_path or list(map(lambda f: f'{os.path.splitext(f)[0]}_OUT.png', input_path))


def get_args() -> argparse.Namespace:
    """defines arguments"""
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='model.pt', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', metavar='scale', type=float, default=0.25,
                        help='Scale factor for the input images')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    in_files = args.input
    out_files = get_output_filenames(args.output, args.input)
    scale = args.scale

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = _get_model()

    for i, filename in enumerate(in_files):
        pred = predict(model, filename)
        result = Image.fromarray(pred).convert('RGB')
        result.save(out_files[i])
