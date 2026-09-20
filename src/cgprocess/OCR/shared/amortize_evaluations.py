import argparse
from pathlib import Path
import numpy as np
import yaml


def process_file(file_path):
    """Read first three rows of a text file, convert to float, return array."""
    with file_path.open("r") as f:
        lines = [float(next(f).strip()) for _ in range(3)]
    return np.array(lines)


def main():
    parser = argparse.ArgumentParser(description="Process txt files in a folder.")
    parser.add_argument(
        "--name",
        "-n",
        type=str,
        help="No default setting for the name, as this is supposed to match the model name.",
    )
    parser.add_argument("--data-path", "-d", type=Path, help="Path to folder containing txt files")
    parser.add_argument(
        "--output-path", "-o",
        type=Path,
        help="Path for output folder."
    )
    args = parser.parse_args()

    folder = args.data_path
    out_path = args.output_path
    out_path.mkdir(parents=True, exist_ok=True)

    if not folder.is_dir():
        print(f"Error: {folder} is not a valid directory")
        return

    all_values = []

    for file_path in folder.glob("*.txt"):
        try:
            values = process_file(file_path)
            all_values.append(values)
        except Exception as e:
            print(f"Skipping {file_path}: {e}")

    if not all_values:
        print("No valid data found.")
        return

    all_values = np.vstack(all_values)
    means = np.mean(all_values, axis=0)
    stds = np.std(all_values, axis=0)
    names = ["Levensthein distance per character", "Perfect lines", "Bad lines"]

    results = {}
    for i, (mean, std) in enumerate(zip(means, stds)):
        print(f"Value {names[i]}: {mean:.4f} ± {std:.4f}")
        results[f"{names[i]}"] = f"{mean:.4f} ± {std:.4f}"

    with open(out_path / f"{args.name}.yml", "w", encoding="utf-8") as file:
        yaml.safe_dump(results, file, allow_unicode=True)


if __name__ == "__main__":
    main()
