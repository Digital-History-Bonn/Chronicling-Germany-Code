"""Module for running lightning trainer."""

import argparse
from multiprocessing import Process, Queue
import multiprocessing
from pathlib import Path
from typing import Optional

import torch
import yaml
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from ssr import Recognizer, SSMOCRTrainer, collate_fn  # pylint: disable=import-error
from torch.utils.data import DataLoader
from torchsummary import summary

from cgprocess.OCR.SSM.dataset import SSMDataset
from cgprocess.OCR.shared.utils import init_tokenizer, load_cfg
from cgprocess.shared.datasets import PageDataset
from cgprocess.shared.multiprocessing_handler import run_processes
from cgprocess.shared.utils import get_file_stem_split


def get_args() -> argparse.Namespace:
    # pylint: disable=duplicate-code
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
        help="Name of the model and the log files.",
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
        "--num-processes",
        type=int,
        default=1,
        help="Number of processes to use for preprocessing.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of threads for loading temp data.",
    )
    parser.add_argument(
        "--eval",
        type=str,
        default=None,
        help="If a model path is provided, this will execute the test run on said model.",
    )
    return parser.parse_args()


def main() -> None:
    """Launch processes for each gpu if possible, otherwise call train directly."""
    args = get_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using {device} device")

    config_path = Path(args.config_path)
    print(f"Model config {config_path}")

    if torch.cuda.is_available() and args.gpus > 1:
        assert torch.cuda.device_count() >= args.gpus, (
            f"More gpus demanded than available! Demanded: "
            f"{args.gpus} Available: {torch.cuda.device_count()}"
        )
        run_multiple_gpus(args)
    else:
        train(args)


def run_multiple_gpus(args: argparse.Namespace) -> None:
    """Launch a process for each gpu."""
    processes = [Process(target=train, args=(args, i)) for i in range(args.gpus)]

    run_processes(
        {"method": get_progress, "args": [args.epochs]},
        processes,
        Queue(),
        args.epochs,
        "Starting",
    )


def get_progress(total: int) -> int:
    """Return total as progress to skip progress bar."""
    return total


def train(args: argparse.Namespace, device_id: Optional[int] = None) -> None:
    """Initialize config, datasets and dataloader and run the lightning trainer."""
    torch.manual_seed(args.seed)
    if device_id is not None:
        torch.manual_seed(args.seed + device_id)
    torch.set_float32_matmul_precision("high")
    data_path = Path(args.data_path)
    config_path = Path(args.config_path)
    # define any number of nn.Modules (or use your current ones)
    cfg = load_cfg(config_path)

    ckpt_dir = Path(f"models/ssm/{args.name}")

    device_id = device_id if device_id else 0

    tokenizer = init_tokenizer(
        cfg
    )  # todo: assertion for wrong vocabulary in saved targets, as well as image height.
    print(f"vocab size: {cfg['vocabulary']['size']}")

    page_dataset = PageDataset(data_path / "images")
    test_file_stems, train_file_stems, val_file_stems = get_file_stem_split(
        args.custom_split_file, args.split_ratio, page_dataset
    )
    if not args.eval:
        kwargs = {
            "data_path": data_path,
            "file_stems": train_file_stems,
            "name": "train",
        }
        train_set = SSMDataset(
            kwargs, cfg["preprocessing"]["image_height"], cfg, augmentation=True, num_processes=args.num_processes,
            augment_params=cfg["training"]["augmentation"], num_threads=args.num_threads
        )  # todo: make shure, this only runs once and not for every gpu
        kwargs = {
            "data_path": data_path,
            "file_stems": val_file_stems,
            "name": "validation",
        }
        val_set = SSMDataset(kwargs, cfg["preprocessing"]["image_height"], cfg, num_threads=args.num_threads)

    kwargs = {"data_path": data_path, "file_stems": test_file_stems, "name": "test"}
    test_set = SSMDataset(kwargs, cfg["preprocessing"]["image_height"], cfg, num_processes=args.num_processes,
                          num_threads=args.num_threads)
    model = Recognizer(cfg).train()
    try:
        print(f"Embedding_size: {model.embedding.weight.shape}")
    except Exception as e:
        print(e)

    summary(model, input_size=(1, 1, 32, 400), batch_dim=0)
    batch_size = args.batch_size

    if not args.eval:
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            prefetch_factor=1,
            persistent_workers=True
        )
        cfg["training"]["steps_per_epoch"] = len(train_loader)
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            prefetch_factor=1,
            persistent_workers=True
        )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        prefetch_factor=1,
        persistent_workers=True,
    )

    lit_model = SSMOCRTrainer(model, batch_size, tokenizer, cfg["training"])
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_levenshtein",
        dirpath=ckpt_dir,
        filename=f"{device_id}-{{epoch}}",
    )

    logger = TensorBoardLogger(f"logs/{args.name}", name=f"{device_id}")
    trainer = Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=1)],
        logger=logger,
        devices=[device_id],
        val_check_interval=0.5,
        limit_val_batches=1.0,
        enable_progress_bar=True
    )  # type: ignore

    if args.eval:
        eval_path = Path(args.eval)
        cfg = load_cfg(eval_path / f"model_{device_id}.yml")
        model_path = eval_path / cfg["inference"]["model_path"]
        model = Recognizer(cfg).eval()
        lit_model = SSMOCRTrainer.load_from_checkpoint(
            model_path, model=model, tokenizer=tokenizer, batch_size=batch_size, hyper_parameters=cfg["training"]
        )
        trainer.test(lit_model, dataloaders=test_loader)
    else:
        # pylint: disable=possibly-used-before-assignment
        print("fit!")
        trainer.fit(
            model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader
        )
        cfg["inference"]["model_path"] = Path(checkpoint_callback.best_model_path).name
        with open(ckpt_dir / f"model_{device_id}.yml", "w", encoding="utf-8") as file:
            yaml.safe_dump(cfg, file)

        lit_model = SSMOCRTrainer.load_from_checkpoint(
            checkpoint_callback.best_model_path,
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            hyper_parameters=cfg["training"]
        )
        trainer.test(lit_model, dataloaders=test_loader)


if __name__ == "__main__":
    main()
