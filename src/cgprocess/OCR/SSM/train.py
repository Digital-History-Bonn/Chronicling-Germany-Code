"""Module for running lightning trainer."""
import argparse
from pathlib import Path

import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import lightning
import yaml

from src.cgprocess.OCR.SSM.dataset import SSMDataset
from ssr import SSMOCRTrainer, Recognizer, collate_fn

from src.cgprocess.OCR.shared.tokenizer import OCRTokenizer


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

    return parser.parse_args()


def main():
    args = get_args()
    data_path = Path(args.data_path)
    # define any number of nn.Modules (or use your current ones)
    with open('mamba_ocr.yml', 'r') as file:
        cfg = yaml.safe_load(file)

    train_set, val_set, test_set = SSMDataset(data_path, cfg).random_split((0.85, 0.05, 0.1))
    model = Recognizer(cfg, train_set.tokenizer)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {DEVICE} device")
    lit_model = SSMOCRTrainer(model)


    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, drop_last=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=8, shuffle=False, drop_last=True, collate_fn=collate_fn)
    checkpoint_callback = ModelCheckpoint(save_top_k=2, monitor="val_loss")
    trainer = lightning.Trainer(max_epochs=50, callbacks=[checkpoint_callback])
    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(checkpoint_callback.best_model_path)
    trainer.test(lit_model, dataloaders=test_loader)


if __name__ == '__main__':
    main()