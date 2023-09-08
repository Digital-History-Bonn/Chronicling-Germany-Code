from time import time
import multiprocessing as mp
import argparse
from src.news_seg.news_dataset import NewsDataset
from src.news_seg.preprocessing import Preprocessing, CROP_SIZE, CROP_FACTOR
from torch.utils.data import DataLoader

EPOCHS = 1
BATCH_SIZE = 64


def test_workers():
    preprocessing = Preprocessing(scale=1, crop_factor=args.crop_factor, crop_size=args.crop_size)
    dataset = NewsDataset(preprocessing, image_path=f"{args.data_path}images/",
                          target_path=f"{args.data_path}targets/",
                          limit=args.limit, dataset=args.dataset)
    for num_workers in range(2, mp.cpu_count()*2, 2):
        train_loader = DataLoader(dataset, shuffle=True, num_workers=num_workers, batch_size=args.batch_size,
                                  pin_memory=True)
        start = time()
        for _ in range(1, args.epochs):
            for i, data in enumerate(train_loader, 0):
                pass
        end = time()
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))


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
    test_workers()
