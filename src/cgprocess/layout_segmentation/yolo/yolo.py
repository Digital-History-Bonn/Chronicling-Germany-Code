import argparse

import torch
from ultralytics import YOLO


def main():
    args = get_args()

    # get weights or params
    if args.seed == 0:
        model_file = {"detect": "yolov8n.pt",
                      "segment": "yolov8n-seg.pt",
                      "pose": "yolov8n-pose.pt"}[args.task]
    else:
        model_file = {"detect": "yolov8n.yaml",
                      "segment": "yolov8n-seg.yaml",
                      "pose": "yolov8n-pose.yaml"}[args.task]

    # Init a model
    model = YOLO(model_file)

    # Train the model
    yaml = {"detect": "data/YOLO_Layout/CGD.yaml",
            "segment": "data/YOLO_Textlines/CGD.yaml",
            "pose": "data/YOLO_Baselines/CGD.yaml"}[args.task]

    imgz = {"detect": 2048,
            "segment": 1024,
            "pose": 1024}[args.task]

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
        "--task",
        "-t",
        type=str,
        default="detect",
        help="task to train must be \"detect\" or \"segment\" or \"pose\""
    )

    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=500,
        help="Number of epochs to train."
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