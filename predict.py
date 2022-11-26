"""Module for loading Model and predicting images"""
import argparse
import os
from typing import Union, List, Any

import numpy as np
from PIL import Image  # type: ignore
import torch  # type: ignore

from model import DhSegment

IN_CHANNELS, OUT_CHANNELS = 3, 10


def _get_model(path: str, device: torch._C.device) -> DhSegment:
    # create model
    model: DhSegment = DhSegment([3, 4, 6, 4], in_channels=IN_CHANNELS, out_channel=OUT_CHANNELS)
    model = model.float()
    model.load(path)
    model.to(device)
    model.eval()
    return model


def get_file(file: str, scaling=0.25) -> torch.tensor:
    """
    loads a image as tensor
    :param file: path to file
    :param scaling: scale
    :return: image as torch.tensor
    """
    img = Image.open(file).convert('RGB')
    shape = int(img.size[0] * scaling), int(img.size[1] * scaling)
    img = img.resize(shape, resample=Image.BICUBIC)

    w_pad, h_pad = (32 - (shape[0] % 32)), (32 - (shape[1] % 32))
    img_np = np.pad(np.array(img), ((0, h_pad), (0, w_pad), (0, 0)), 'constant', constant_values=0)
    img_t = np.transpose(torch.tensor(img_np), (2, 0, 1))
    return torch.unsqueeze(img_t, dim=0)


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

    model = _get_model(args.model, DEVICE)

    for i, filename in enumerate(in_files):
        image = get_file(filename, scaling=scale)
        pred = np.array(model.predict(torch.tensor(image).to(DEVICE)))
        result = Image.fromarray(pred).convert('RGB')
        result.save(out_files[i])
