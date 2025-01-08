"""Scipt for extracting random lines from one file. E.g. to select random files from a list of files. 
Extract file list through sftp: ' echo ls -f -1 | sftp HOST:/FOLDER > data/file_list.txt'"""
import random
import argparse
import re
from pathlib import Path

def select_random_lines(input_file: Path, output_file: Path, num_lines: int, file_transfer: str):
    try:
        # Read all lines from the input file
        print(f"loading input file {input_file}, please be patient.")
        with open(input_file, 'r') as infile:
            lines = infile.readlines()

        # Check if the file has fewer lines than the requested number
        if len(lines) < num_lines:
            print(f"Warning: The file contains only {len(lines)} lines. Selecting all of them.")
            selected_lines = lines
        else:
            # Randomly select the specified number of lines
            print("sampling random lines")
            selected_lines = random.sample(lines, num_lines)

        result_list = []
        if file_transfer:
            result_list = [f"sftp {file_transfer} <<EOF \n"]
            for line in selected_lines:
                line = re.sub(' +', ' ', line)
                for file in line.split():
                    if file[-5:] == ".jpeg" or file[-4:] == ".jpg":
                        result_list.append(f"get {file}\n")

        print(f"writing result to output file  {output_file}.")
        # Write the selected lines to the output file
        with open(output_file, 'w') as outfile:
            outfile.writelines(result_list)

        print(f"{len(result_list)} lines have been written to {output_file}.")

    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# pylint: disable=locally-disabled, duplicate-code
def get_args() -> argparse.Namespace:
    """defines arguments"""
    parser = argparse.ArgumentParser(description="Extract a random amount of lines from a given file and save it to a new file.")
    parser.add_argument(
        "--input-path",
        "-i",
        type=str,
        default="data/",
        help="Input path for lines file.",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        type=str,
        default="output/",
        help="Output path for random lines file.",
    )
    parser.add_argument(
        "--num_lines",
        "-n",
        type=int,
        default=100,
        help="Number of lines that should be selected randomly.",
    )
    parser.add_argument(
        "--file_transfer",
        "-f",
        type=str,
        default=None,
        help="Supply host name and folder for ftp connection. Lines will be split between spaces, as well, as only results with .jpeg extension are valid. Furthermore, the output file will be an .sh file to load all those files from an ftp server. If left empty, this option is deactivated.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    select_random_lines(Path(args.input_path), Path(args.output_path), num_lines=args.num_lines, args.file_transfer)
