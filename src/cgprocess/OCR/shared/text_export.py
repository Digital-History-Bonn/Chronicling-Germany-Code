"""Export text acording to layout information."""

import argparse
import json
import os
import re
import warnings
from multiprocessing import Process, Queue
from pathlib import Path
from time import sleep
from typing import Tuple, List, Dict

import numpy as np
from numpy import ndarray
from bs4 import BeautifulSoup, ResultSet
from tqdm import tqdm

from src.cgprocess.OCR.shared.merge_text_export import merge_files
from src.cgprocess.OCR.shared.utils import line_has_text


# todo: tests
def extract_text(xml_data: BeautifulSoup, path: str, export_lines: bool) -> Tuple[
    ndarray, list]:
    """
    Sorts regions and lines by the supplied reading order and assembles ndarray for csv export and
    dictionary for json export.
    """
    text_regions = xml_data.find_all("TextRegion")
    if len(text_regions) == 0:
        return np.array([]), []

    region_order = sort_xml_elements(text_regions)

    region_csv: list = []
    region_list: list = []
    for i, region_id in enumerate(region_order):
        region = xml_data.find(attrs={'id': f'{region_id}'})
        lines = region.find_all("TextLine")
        if len(lines) == 0:
            continue
        lines_order = sort_xml_elements(lines)

        region_text_list = []
        region_confidence_list = []
        for _, line_id in enumerate(lines_order):
            line = xml_data.find(attrs={'id': f'{line_id}'})
            if line_has_text(line): # type: ignore
                region_text_list.append(line.TextEquiv.Unicode.contents[0])
                region_confidence_list.append(line.TextEquiv.attrs["conf"])

        region_type = re.search(
            r"structure \{type:(.+?);}", region["custom"]
        )
        region_class = "UnkownRegion" if not region_type else region_type.group(1)

        if export_lines:
            prepare_csv_data(i, path, region_class, region_csv, region_text_list, region_confidence_list)
        else:
            region_csv.append([path, str(i), region_class,
                               np.mean(np.array(region_confidence_list, dtype=float)), "\n".join(region_text_list)])

        # todo: add confidence to json
        if export_lines:
            region_list.append({"class": region_class, "lines": region_text_list})
        else:
            region_list.append({"class": region_class, "text": region_text_list})

    if len(region_csv) == 0:
        return np.array([]), region_list
    return np.vstack(region_csv), region_list


def prepare_csv_data(i: int, path: str, region_class: str, region_csv: list, region_text_list: List[str],
                     region_confidence_list: List[str]
                     ) -> None:
    """
    Assemble csv data for a single region. This means adding the current region id to all lines of that
    region, as well ass the class and page path.
    :param region_csv: csv array for this page, that contains ndarrays for every region.
    """
    path_array = np.full(len(region_text_list), path)
    region_id_array = np.full(len(region_text_list), str(i))
    class_array = np.full(len(region_text_list), region_class)
    line_id_array = np.char.mod('%d', (np.arange(len(region_text_list)) + 1))
    region_csv.append(
        np.vstack([path_array, region_id_array, line_id_array, class_array, np.array(region_confidence_list),
                   np.array(region_text_list)]).T)


def sort_xml_elements(elements: ResultSet) -> ndarray:
    """
    Sort xml elements by readingOrder.
    :param elements: regions or lines list.
    """
    reading_list = []
    for element in elements:

        reading = re.search(
            r"readingOrder \{index:(.+?);}", element["custom"]
        )

        if reading:
            reading_list.append([int(reading.group(1)), element["id"]])
        else:
            warnings.warn("No reading Order found. This line will be ignored.")
    order = np.array(reading_list)
    if len(reading_list) < 1:
        return np.array([])
    return order[:, 1][np.argsort(order[:, 0])]  # type: ignore


def main(args: argparse.Namespace) -> None:
    """
    Load xml files and assemble page lists before saving them.
    """
    data_path = Path(args.data_path)
    output_path = Path(args.output_path)

    if not os.path.exists(output_path):
        print(f"creating {output_path}")
        os.makedirs(output_path)

    if not os.path.exists(output_path / "temp"):
        print(f"creating {output_path / 'temp'}")
        os.makedirs(output_path / "temp")

    paths = [
        f[:-4] for f in os.listdir(data_path) if f.endswith(".xml")
    ]

    total = len(paths)

    launch_processes(args, data_path, output_path, paths, total)

    if not args.skip_merge:
        # todo: add json handling
        paths = [
            f[:-4] for f in os.listdir(output_path / "temp") if f.endswith(".npz")
        ]
        merge_files(args, output_path, paths)


def launch_processes(args: argparse.Namespace, data_path: Path, output_path: Path, paths: list, total: int):
    """Launch processes for text extraction."""
    # todo: integrate this into multiprocessing handler
    path_queue: Queue = Queue()
    processes = [Process(target=run_text_extraction,
                         args=(data_path, output_path, path_queue, args.lines, {"csv": args.csv, "json": args.json}))
                 for _ in range(args.process_count)]
    for process in processes:
        process.start()
    counter = total
    with tqdm(total=total, desc="Read xml data", unit="pages") as pbar:
        while not path_queue.empty() or counter > 0:
            for _ in range(1000):
                if counter == 0:
                    break
                path_queue.put((paths[counter - 1], False), True)
                counter -= 1
                pbar.n = total - path_queue.qsize() - counter
                pbar.refresh()

            pbar.n = total - path_queue.qsize() - counter
            pbar.refresh()
            sleep(1)
    for _ in processes:
        path_queue.put(("", True))
    for process in tqdm(processes, desc="Waiting for processes to end"):
        process.join()


def run_text_extraction(data_path: Path, output_path: Path, path_queue: Queue, lines: bool,
                        export: Dict[str, bool]) -> None:
    """Get path from MP Queue and save data to npz or json file."""
    while True:
        path, done = path_queue.get()
        if done:
            break
        with open(data_path / f"{path}.xml", "r", encoding="utf-8") as file:
            data = file.read()
        xml_data = BeautifulSoup(data, "xml")
        region_csv, region_json = extract_text(xml_data, path, lines)
        if (len(region_csv) == 0 and export["csv"]) or (len(region_json) == 0 and export["json"]):
            print(f"{path}.xml contains no text")
            continue
        if export["csv"]:
            np.savez_compressed(output_path / "temp" / f"{path}.npz", array=region_csv)
        if export["json"]:
            with open(output_path / "temp" / f"{path}.json", 'w', newline='', encoding='utf-8') as file:
                json.dump(region_json, file)


def get_args() -> argparse.Namespace:
    # pylint: disable=duplicate-code
    """defines arguments"""
    parser = argparse.ArgumentParser(description="predict")
    parser.add_argument(
        "--data-path",
        "-d",
        type=str,
        default="data/lines/",
        help="path for input xml folder",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        type=str,
        dest="output_path",
        default="data/output-text/",
        help="path for output folder",
    )
    parser.add_argument(
        "--lines",
        action="store_true",
        help="Activates export of lines individually, instead of combining lines to article text.",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Activates csv export.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Activates json export.",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Deactivates merging",
    )
    parser.add_argument(
        "--process-count",
        "-p",
        type=int,
        default=1,
        help="Select number of processes that are launched.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    parameter_args = get_args()
    assert parameter_args.csv or parameter_args.json, ("Please activate at least one export method with '--csv' or "
                                                       "'--json'.")
    main(parameter_args)
