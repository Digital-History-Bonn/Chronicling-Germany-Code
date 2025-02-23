"""Module for running lightning trainer."""
import argparse
import json
import os
from multiprocessing import set_start_method, Value, Process, Queue
from multiprocessing.sharedctypes import Synchronized
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
from PIL import Image
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer
from torchsummary import summary
from ssr import SSMOCRTrainer, Recognizer, collate_fn

from src.cgprocess.OCR.SSM.dataset import SSMDataset
from src.cgprocess.OCR.shared.utils import load_cfg, init_tokenizer
from src.cgprocess.shared.datasets import PageDataset
from src.cgprocess.shared.multiprocessing_handler import run_processes
from src.cgprocess.shared.utils import get_file_stem_split


def get_args() -> argparse.Namespace:
    """
    Defines arguments.

    Returns:
        Namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train SSM OCR")

    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=10,
        help="number of epochs to train",
    )
    parser.add_argument(
        "--name",
        "-n",
        type=str,
        default=None,
        help="Name of the model and the log files."
    )
    # pylint: disable=duplicate-code
    parser.add_argument(
        "--data-path",
        "-d",
        type=str,
        default=None,
        help="path for folder with folders 'images' and 'annotations'",
    )

    # pylint: disable=duplicate-code
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="Seeding number for random generators.",
    )
    parser.add_argument(
        "--custom-split-file",
        type=str,
        default=None,
        help="Provide path for custom split json file. This should contain a list with file stems "
             "of train, validation and test images. File stem is the file name without the extension.",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        nargs="+",
        default=(0.85, 0.05, 0.10),
        help="Takes 3 float values for a custom dataset split ratio. The ratio have to sum up to one and the Dataset "
             "has to be big enough, to contain at least one batch for each dataset. Provide ratios for train, test "
             "and validation in this order.",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        dest="batch_size",
        metavar="B",
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "--num-workers",
        "-w",
        type=int,
        default=1,
        help="Number of workers for the Dataloader",
    )
    parser.add_argument(
        "--config_path",
        "-cp",
        type=str,
        default="config/cfg.yml",
        help="Path to model config.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="If cuda is available, this determines the number of processes launched, each receiving a single gpu.",
    )
    parser.add_argument(
        "--eval",
        type=str,
        default=None,
        help="If a model path is provided, this will execute the test run on said model.",
    )
    return parser.parse_args()


def main():
    args = get_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using {device} device")

    config_path = Path(args.config_path)
    print(f"Model config {config_path}")

    if torch.cuda.is_available() and args.gpus > 1:
        assert torch.cuda.device_count() >= args.gpus, f"More gpus demanded than available! Demanded: {args.gpus} Available: {torch.cuda.device_count()}"
        run_multiple_gpus(args)
    else:
        train(args)


def run_multiple_gpus(args: argparse.Namespace) -> None:
    processes = [Process(target=train,
                         args=(args, i)) for i in range(args.gpus)]

    run_processes({"method": get_progress, "args": [args.epochs]}, processes, Queue(), args.epochs,
                  "Starting")


def get_progress(total: int):
    return total


def train(args: argparse.Namespace, device_id: Optional[int] = None) -> None:
    torch.set_float32_matmul_precision('high')
    data_path = Path(args.data_path)
    config_path = Path(args.config_path)
    # define any number of nn.Modules (or use your current ones)
    cfg = load_cfg(config_path)

    ckpt_dir = Path(f'models/ssm/{args.name}')

    device_id = device_id if device_id else 0

    tokenizer = init_tokenizer(cfg)  # todo: assertion for wrong vocabulary in saved targets.
    print(f"vocab size: {cfg['vocabulary']['size']}")

    page_dataset = PageDataset(data_path / "images")
    test_file_stems, train_file_stems, val_file_stems = get_file_stem_split(args.custom_split_file,
                                                                            args.split_ratio,
                                                                            page_dataset)
    if not args.eval:
        kwargs = {"data_path": data_path, "file_stems": train_file_stems, "name": "train"}
        train_set = SSMDataset(kwargs, cfg["image_height"], cfg, augmentation=False, num_processes=1)
        kwargs = {"data_path": data_path, "file_stems": val_file_stems, "name": "validation"}
        val_set = SSMDataset(kwargs, cfg["image_height"], cfg)
    kwargs = {"data_path": data_path, "file_stems": test_file_stems, "name": "test"}
    test_set = SSMDataset(kwargs, cfg["image_height"], cfg)
    model = Recognizer(cfg).train()

    summary(model, input_size=(1, 1, 32, 400), batch_dim=0)
    batch_size = args.batch_size

    lit_model = SSMOCRTrainer(model, batch_size, tokenizer)

    if not args.eval:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn,
                                  num_workers=args.num_workers,
                                  prefetch_factor=2,
                                  persistent_workers=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn,
                                num_workers=args.num_workers,
                                prefetch_factor=2,
                                persistent_workers=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn,
                             num_workers=args.num_workers,
                             prefetch_factor=2,
                             persistent_workers=True)

    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_levenshtein", dirpath=ckpt_dir,
                                          filename=f'{device_id}-{{epoch}}')

    logger = TensorBoardLogger(f"logs/{args.name}", name=f"{device_id}")
    trainer = Trainer(max_epochs=args.epochs, callbacks=[checkpoint_callback], logger=logger,
                      devices=[device_id], val_check_interval=0.5, limit_val_batches=0.5)  # type: ignore

    if args.eval:
        eval_path = Path(args.eval)
        model_path = eval_path / [f for f in os.listdir(eval_path) if f.startswith(f"{device_id}")][0]
        model = Recognizer(cfg).eval()
        lit_model = SSMOCRTrainer.load_from_checkpoint(model_path, model=model, tokenizer=tokenizer,
                                                       batch_size=batch_size)
        trainer.test(lit_model, dataloaders=test_loader)
    else:
        trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        lit_model = SSMOCRTrainer.load_from_checkpoint(checkpoint_callback.best_model_path, model=model,
                                                       tokenizer=tokenizer,
                                                       batch_size=batch_size)
        trainer.test(lit_model, dataloaders=test_loader)


if __name__ == '__main__':
    main()
