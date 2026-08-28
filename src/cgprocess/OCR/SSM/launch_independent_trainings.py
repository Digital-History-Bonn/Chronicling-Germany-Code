import argparse
from datetime import datetime
import subprocess
from pathlib import Path

import torch

from train import get_parser


def get_args() -> argparse.Namespace:
    parser = get_parser()

    parser.add_argument(
        "--gpus",
        type=int,
        default=4,
        help="If cuda is available, this determines the number of processes launched, each receiving a single gpu.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=f"logs/{datetime.now().strftime('%A, %B %d, %Y')}",
        help="If cuda is available, this determines the number of processes launched, each receiving a single gpu.",
    )
    parser.add_argument(
        "--script-path",
        type=Path,
        default=f"src/cgprocess/OCR/SSM/train.py",
        help="If cuda is available, this determines the number of processes launched, each receiving a single gpu.",
    )
    return parser.parse_args()


def main():
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
    else:
        args.gpus = 1

    print(f"Supplied Arguments: {namespace_to_argv(args)}")
    print("\n\n\n")

    log_path = args.log_path
    del args.log_path

    num_gpus = args.gpus
    del args.gpus

    script_path = args.script_path
    del args.script_path

    assert script_path.is_file() and script_path.suffix == ".py", f"Script path {script_path} is not valid. Please supply a correct path to a python script."
    log_path.mkdir(parents=True, exist_ok=True)
    print(f"Log path: {log_path.absolute()}")

    processes = []
    for i in range(num_gpus):
        log_out_path = log_path / f"{i}.out"
        log_err_path = log_path / f"{i}.err"
        log_out_file = log_out_path.open("w", encoding="utf-8", buffering=1)
        log_err_file = log_err_path.open("w", encoding="utf-8", buffering=1)

        args.gpu_id = i

        processes.append(subprocess.Popen(
            ["python", script_path, *namespace_to_argv(args)],
            stdout=log_out_file,
            stderr=log_err_file,
            text=True
        ))
        print(f"Launch Training on cuda:{i}")

    for i, process in enumerate(processes):
        return_code = process.wait()
        log_out_file.close()
        log_err_file.close()

        print(f"Training on GPU {i} exited with code {return_code}")


def namespace_to_argv(args):
    command = []

    for name, value in vars(args).items():
        option = "--" + name.replace("_", "-")

        if isinstance(value, bool):
            if value:
                command.append(option)
        elif value is not None:
            command.extend([option, str(value)])

    return command


if __name__ == "__main__":
    main()
