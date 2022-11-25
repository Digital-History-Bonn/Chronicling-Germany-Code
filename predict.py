"""Module for loading Model and predicting images"""
import argparse
import os
from typing import Union, List, Any

import numpy as np
from PIL import Image  # type: ignore
import torch
from torchvision import transforms  # type: ignore

from model import DhSegment

IN_CHANNELS, OUT_CHANNELS = 3, 10


def get_output_filenames(output_path: Union[str, List[Any]], input_path: Union[str, List[Any]]) -> Union[
    str, List[Any]]:
    """returns generated output name or output name from args"""

    def _generate_name(function):
        return f'{os.path.splitext(function)[0]}_OUT.png'

    return output_path or list(map(_generate_name, input_path))


def get_args() -> argparse.Namespace:
    """defines arguments"""
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='model.pt', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.25,
                        help='Scale factor for the input images')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    in_files = args.input
    out_files = get_output_filenames(args.output, args.input)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # DEVICE = 'cpu'

    # create model
    model = DhSegment([3, 4, 6, 4], in_channels=IN_CHANNELS, out_channel=OUT_CHANNELS, load_resnet_weights=True)
    model = model.float()
    model.load(args.model)
    model.to(DEVICE)
    model.eval()

    transform = transforms.Compose([
        transforms.PILToTensor()
    ])

    for i, filename in enumerate(in_files):
        img = Image.open(filename).convert('RGB')
        # shape = int(img.size[0] * args.scale), int(img.size[1] * args.scale)
        shape = int(1024), int(1024)

        img = img.resize(shape, resample=Image.BICUBIC)
        pred = model(torch.unsqueeze(transform(img), dim=0).to(DEVICE)).argmax(dim=1).float().cpu()

        result = Image.fromarray(np.array(torch.squeeze(pred / OUT_CHANNELS)) * 255).convert('RGB')
        result.save(out_files[i])
