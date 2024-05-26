"""Module for importing xml files and updating the reading order of regions and lines. TODO: lines"""
import argparse
import os
from typing import List, Dict

from tqdm import tqdm

from src.news_seg.processing.read_xml import read_raw_data, read_regions_for_reading_order
from src.news_seg.processing.reading_order import PageProperties
from src.news_seg.processing.transkribus_export import copy_xml
from src.news_seg.utils import adjust_path


def align_ids(id_dict: Dict[int, List[str]]) -> List[str]:
    """Collapses list of region lists(2d array) to a single list ids (1d)."""
    result = []
    for _, id_list in id_dict.items():
        for region_id in id_list:
            result.append(region_id)
    return result


def main(parsed_args: argparse.Namespace) -> None:
    """Handles loading of xml files and updating the reading order of regions."""
    data_paths = [
        f[:-4] for f in os.listdir(parsed_args.data_path) if f.endswith(".xml")
    ]

    output_path = adjust_path(parsed_args.output_path)
    data_path = adjust_path(parsed_args.data_path)
    if not os.path.exists(output_path):
        print(f"creating {output_path}.")
        os.makedirs(output_path)

    for path in tqdm(data_paths):
        bbox_dict, id_dict, bs_data = read_regions_for_reading_order(f"{data_path}{path}")
        bs_copy = read_raw_data(f"{data_path}{path}")
        if len(bbox_dict) == 0:
            continue
        page = PageProperties(bbox_dict)
        reading_order_dict = page.get_reading_order()
        # with open(f"{output_path}{path}.json", "w", encoding="utf-8") as file:
        #     json.dump(reading_order_dict, file)

        id_list = align_ids(id_dict)

        copy_xml(bs_copy, bs_data, id_list, reading_order_dict)

        with open(
                f"{output_path}{path}.xml",
                "w",
                encoding="utf-8",
        ) as xml_file:
            xml_file.write(
                bs_copy.prettify().replace("<Unicode>\n      ", "<Unicode>").replace("\n     </Unicode>", "</Unicode>"))


def get_args() -> argparse.Namespace:
    """defines arguments"""
    # pylint: disable=locally-disabled, duplicate-code
    parser = argparse.ArgumentParser(description="creates targets from annotation xmls")
    parser.add_argument(
        "--data-path",
        "-d",
        type=str,
        dest="data_path",
        default="data/",
        help="path for folder with xml files.",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        type=str,
        dest="output_path",
        default="output/",
        help="path for output folder",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
