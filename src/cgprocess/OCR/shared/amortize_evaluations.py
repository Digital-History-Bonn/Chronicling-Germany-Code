import argparse
from pathlib import Path
import numpy as np


def process_file(file_path):
    """Read first three rows of a text file, convert to float, return array."""
    with file_path.open("r") as f:
        lines = [float(next(f).strip()) for _ in range(3)]
    return np.array(lines)


def main():
    parser = argparse.ArgumentParser(description="Process txt files in a folder.")
    parser.add_argument("--data-path", "-d", type=str, help="Path to folder containing txt files")
    args = parser.parse_args()

    folder = Path(args.data_path)

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

    for i, (mean, std) in enumerate(zip(means, stds)):
        print(f"Value {names[i]}: {mean:.4f} ± {std:.4f}")


if __name__ == "__main__":
    main()
