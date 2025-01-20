"""Model to train kraken OCR model."""
import argparse
import glob
import os
from pprint import pprint

import torch
from kraken.lib import default_specs                            # pylint: disable=no-name-in-module, import-error
from kraken.lib.train import RecognitionModel, KrakenTrainer    # pylint: disable=no-name-in-module, import-error
from lightning.pytorch.loggers import TensorBoardLogger         # pylint: disable=import-error

from src.cgprocess.OCR.shared.utils import set_seed, adjust_path

torch.set_float32_matmul_precision('medium')


def get_args() -> argparse.Namespace:
    """
    Defines arguments.

    Returns:
        Namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser(description="train kraken OCR")

    parser.add_argument(
        "--name",
        "-n",
        type=str,
        default=None,
        help="Name of the model and the log files."
    )
    # pylint: disable=duplicate-code
    parser.add_argument(
        "--train_data",
        "-t",
        type=str,
        default=None,
        help="path for folder with images jpg files and annotation xml files to train the model."
    )

    parser.add_argument(
        "--valid_data",
        "-v",
        type=str,
        default=None,
        help="path for folder with images jpg files and annotation xml files to validate the model."
    )

    # pylint: disable=duplicate-code
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="Seeding number for random generators.",
    )

    return parser.parse_args()


def main() -> None:
    """Trains a OCR model."""
    # get args
    args = get_args()
    set_seed(args.seed)
    print(f"{args =}")

    name = args.name
    train_path = adjust_path(args.train_data)
    valid_path = adjust_path(args.valid_data)

    # check for cuda device
    if torch.cuda.is_available():
        print("CUDA device is available!")
    else:
        print("CUDA device is not available.")

    # create folder for model saves
    os.makedirs(f'models/{name}', exist_ok=False)

    # create training- and evaluation set
    training_files = list(glob.glob(f"{train_path}/*.xml"))
    evaluation_files = list(glob.glob(f"{valid_path}/*.xml"))

    print(f"{len(training_files)} training images and {len(evaluation_files)} validation images.")

    # set some hyperparameters
    hparams = default_specs.RECOGNITION_HYPER_PARAMS.copy()
    hparams['epochs'] = 5
    hparams['lrate'] = 0.001
    hparams['warmup'] = 1
    hparams['augment'] = True
    hparams['batch_size'] = 32  # <- 32

    # init model
    model = RecognitionModel(hyper_params=hparams,
                             output=f'models/{name}/model',
                             # model='load_model/german_newspapers_kraken.mlmodel',
                             num_workers=16,
                             training_data=training_files,
                             evaluation_data=evaluation_files,
                             resize='new',
                             format_type='page')

    # print hyperparameter of model
    pprint(f"{model.hparams}")

    # init logger and training
    logger = TensorBoardLogger("logs", name=name)
    trainer = KrakenTrainer(pl_logger=logger)  # pylint: disable=unexpected-keyword-arg, no-value-for-parameter

    # start training
    trainer.fit(model)  # pylint: disable=no-member


if __name__ == '__main__':
    main()
