"""Module for running lightning trainer."""
import argparse
from pathlib import Path

import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import lightning
from pytorch_lightning import loggers as pl_loggers
from torchsummary import summary
from ssr import SSMOCRTrainer, Recognizer, collate_fn

from src.cgprocess.OCR.SSM.dataset import SSMDataset
from src.cgprocess.OCR.shared.utils import load_cfg, init_tokenizer
from src.cgprocess.shared.datasets import PageDataset
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

    return parser.parse_args()


def main():
    torch.set_float32_matmul_precision('high')

    args = get_args()
    data_path = Path(args.data_path)
    config_path = Path(args.config_path)
    # define any number of nn.Modules (or use your current ones)
    cfg = load_cfg(config_path)

    tokenizer = init_tokenizer(cfg)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {DEVICE} device")
    print(f"Model config {config_path}")

    page_dataset = PageDataset(data_path / "images")
    test_file_stems, train_file_stems, val_file_stems = get_file_stem_split(args.custom_split_file, args.split_ratio,
                                                                            page_dataset)
    kwargs = {"data_path": data_path, "file_stems": train_file_stems, "name": "train"}
    train_set = SSMDataset(kwargs, cfg["image_height"], cfg, augmentation=True)
    kwargs = {"data_path": data_path, "file_stems": val_file_stems, "name": "validation"}
    val_set = SSMDataset(kwargs, cfg["image_height"], cfg)
    kwargs = {"data_path": data_path, "file_stems": test_file_stems, "name": "test"}
    test_set = SSMDataset(kwargs, cfg["image_height"], cfg)
    model = Recognizer(cfg).train()

    summary(model, input_size=(1, 1, 32, 400), batch_dim=0)
    batch_size = args.batch_size

    lit_model = SSMOCRTrainer(model, batch_size, tokenizer)

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
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss", dirpath=f'models/ssm/{args.name}',
                                          filename=f'{{epoch}}-{{val_loss:.2f}}-')

    trainer = lightning.Trainer(max_epochs=args.epochs, callbacks=[checkpoint_callback], devices=1)
    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(checkpoint_callback.best_model_path)
    trainer.test(lit_model, dataloaders=test_loader)


if __name__ == '__main__':
    main()
