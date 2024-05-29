"""Model to train kraken OCR model."""
import argparse
import glob
import os
from pprint import pprint

import torch
from kraken.lib import default_specs                            # pylint: disable=no-name-in-module
from kraken.lib.train import RecognitionModel, KrakenTrainer    # pylint: disable=no-name-in-module
from lightning.pytorch.loggers import TensorBoardLogger

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

    return parser.parse_args()


def main() -> None:
    """Trains a OCR model."""

    # get args
    args = get_args()
    name = args.name

    train_path = args.train_data
    valid_path = args.valid_data

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
    hparams['epochs'] = 100
    hparams['lrate'] = 0.001
    hparams['warmup'] = 1
    hparams['augment'] = True
    hparams['batch_size'] = 128
    hparams['freeze_backbone'] = 2

    # init model
    model = RecognitionModel(hyper_params=hparams,
                             output=f'models/{name}/model',
                             # model='load_model/german_newspapers_kraken.mlmodel',
                             num_workers=23,
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
