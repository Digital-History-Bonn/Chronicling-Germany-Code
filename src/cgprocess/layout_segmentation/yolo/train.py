import argparse

import torch
from ultralytics import YOLO


def main():
    args = get_args()

    # get weights or params
    model_file = "yolov8n.pt" if args.seed == 0 else "yolov8n.yaml"

    # Init a model
    model = YOLO(model_file)

    # Train the model
    yaml = f"{args.data}/CGD.yaml"
    imgz = 2048

    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else 'cpu'

    model.train(name=args.name,
                data=yaml,
                epochs=args.epochs,
                patience=args.epochs,
                imgsz=imgz,
                batch=8 * len(devices) if torch.cuda.is_available() else 8,
                device=devices,
                seed=args.seed,
                cache=True)


def get_args() -> argparse.Namespace:
    """
    Defines arguments.

    Returns:
        Namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser(description="predict")
    # pylint: disable=duplicate-code
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=500,
        help="Number of epochs to train."
    )

    parser.add_argument(
        "--data",
        "-d",
        type=str,
        help="Path to the preprocessed data."
    )

    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=0,
        help="Seed for training. If seed is not set or 0 pretrained weights are used."
    )

    parser.add_argument(
        "--name",
        "-n",
        type=str,
        default=None,
        help="Name of the trained model."
    )

    return parser.parse_args()


if __name__ == '__main__':
    main()