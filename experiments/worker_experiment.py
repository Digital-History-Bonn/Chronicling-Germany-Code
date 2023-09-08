"""Experiment script for number of workers"""
import argparse
import multiprocessing as mp
from time import time  # type: ignore

from torch.utils.data import DataLoader  # type: ignore

from src.news_seg.news_dataset import NewsDataset  # type: ignore
from src.news_seg.preprocessing import Preprocessing, CROP_SIZE, CROP_FACTOR  # type: ignore

EPOCHS = 1
BATCH_SIZE = 64


def worker_experiment():
    """Load data with increasing amount of workers"""
    preprocessing = Preprocessing(scale=1, crop_factor=args.crop_factor, crop_size=args.crop_size)
    dataset = NewsDataset(preprocessing, image_path=f"{args.data_path}images/",
                          target_path=f"{args.data_path}targets/",
                          limit=args.limit, dataset=args.dataset)
    for num_workers in range(2, mp.cpu_count() * 2, 2):
        train_loader = DataLoader(dataset, shuffle=True, num_workers=num_workers, batch_size=args.batch_size,
                                  pin_memory=True)
        start = time()
        for _ in range(1, args.epochs):
            for _, _ in enumerate(train_loader, 0):
                pass
        end = time()
        print(f"Finish with:{end - start} second, num_workers={num_workers}")


# pylint: disable=duplicate-code
def get_args() -> argparse.Namespace:
    """defines arguments"""
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument(
        "--epochs",
        "-e",
        metavar="EPOCHS",
        type=int,
        default=EPOCHS,
        help="number of epochs to train",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        dest="batch_size",
        metavar="B",
        type=int,
        default=BATCH_SIZE,
        help="Batch size",
    )
    parser.add_argument(
        "--data-path",
        "-d",
        type=str,
        dest="data_path",
        default=None,
        help="path for folder with folders 'images' and 'targets'",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="limit quantity of loaded images for testing purposes",
    )
    parser.add_argument('--dataset', type=str, default="transcribus",
                        help="which dataset to expect. Options are 'transcribus' and 'HLNA2013' "
                             "(europeaner newspaper project)")
    parser.add_argument('--crop_size', type=int, default=CROP_SIZE, help='Window size of image cropping')
    parser.add_argument('--crop_factor', type=float, default=CROP_FACTOR, help='Scaling factor for cropping steps')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    worker_experiment()
