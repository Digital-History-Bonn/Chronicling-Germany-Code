"""Module for running lightning trainer."""
import argparse
from pathlib import Path

import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.nn import ConstantPad1d
from torch.utils.data import DataLoader, random_split
import lightning
import yaml
from torchvision import transforms

from src.cgprocess.OCR.SSM.dataset import TrainDataset
from src.cgprocess.OCR.SSM.mamba_recognizer import Recognizer
from src.cgprocess.OCR.SSM.trainer import SSMOCRTrainer

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
        "--image_data",
        "-t",
        type=str,
        default=None,
        help="path for folder with images jpg files and annotation xml files to train the model."
    )

    parser.add_argument(
        "--xml_data",
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


def main():
    args = get_args()
    image_path = Path(args.image_data)
    xml_path = Path(args.xml_data)
    # define any number of nn.Modules (or use your current ones)
    with open('mamba_ocr.yml', 'r') as file:
        cfg = yaml.safe_load(file)
    model = Recognizer(cfg)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {DEVICE} device")
    lit_model = SSMOCRTrainer(model)

    train_set, val_set, test_set = TrainDataset(image_path, xml_path, cfg).random_split((0.85, 0.05, 0.1))
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, drop_last=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=8, shuffle=False, drop_last=True, collate_fn=collate_fn)
    checkpoint_callback = ModelCheckpoint(save_top_k=2, monitor="val_loss")
    trainer = lightning.Trainer(max_epochs=5, callbacks=[checkpoint_callback])
    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(checkpoint_callback.best_model_path)
    trainer.test(lit_model, dataloaders=test_loader)


if __name__ == '__main__':
    main()