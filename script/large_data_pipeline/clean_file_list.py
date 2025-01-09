import argparse
from pathlib import Path


def main(data_path):
    with open(data_path / "file_list.txt", 'r') as infile:
        lines = infile.readlines()
    result_list = []
    for line in lines:
        line = line[:-1]
        if line[-5:] == ".jpeg" or line[-4:] == ".jpg":
            result_list.append(f"{line}\n")
    with open(data_path / "clean_file_list.txt", 'w') as outfile:
        outfile.writelines(result_list)





def get_args() -> argparse.Namespace:
    """defines arguments"""
    parser = argparse.ArgumentParser(
        description="Extract a random amount of lines from a given file and save it to a new file.")
    parser.add_argument(
        "--data-path",
        "-d",
        type=str,
        default="data/",
        help="Data path for where input file 'file_list.txt' will be read and output file "
             "'clean_file_list.txt' will be saved.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(Path(args.data_path))