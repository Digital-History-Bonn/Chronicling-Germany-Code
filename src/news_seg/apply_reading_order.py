"""Module for importing xml files and updating the reading order of regions and lines. TODO: lines"""
import argparse
import os
import re
from typing import List, Dict

from bs4 import BeautifulSoup
from tqdm import tqdm

from src.news_seg.processing.read_xml import read_raw_data, read_regions_for_reading_order
from src.news_seg.processing.reading_order import PageProperties


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

    if not os.path.exists(parsed_args.output_path):
        print(f"creating {parsed_args.output_path}.")
        os.makedirs(parsed_args.output_path)

    for path in tqdm(data_paths):
        bbox_dict, id_dict, bs_data = read_regions_for_reading_order(f"{parsed_args.data_path}{path}")
        bs_copy = read_raw_data(f"{parsed_args.data_path}{path}")
        if len(bbox_dict) == 0:
            continue
        page = PageProperties(bbox_dict)
        reading_order_dict = page.get_reading_order()
        # with open(f"{parsed_args.output_path}{path}.json", "w", encoding="utf-8") as file:
        #     json.dump(reading_order_dict, file)

        id_list = align_ids(id_dict)

        create_xml(bs_copy, bs_data, id_list, reading_order_dict)

        with open(
                f"{parsed_args.output_path}{path}.xml",
                "w",
                encoding="utf-8",
        ) as xml_file:
            xml_file.write(
                bs_copy.prettify().replace("<Unicode>\n      ", "<Unicode>").replace("\n     </Unicode>", "</Unicode>"))


def create_xml(bs_copy: BeautifulSoup, bs_data: BeautifulSoup, id_list: List[str],
               reading_order_dict: dict[int, int]) -> None:
    """
    Copy regions into new BeautifulSoup object with corrected reading order.
    """
    page = bs_copy.find("Page")
    page.clear()
    order_group = bs_copy.new_tag(
        "OrderedGroup", attrs={"caption": "Regions reading order"}
    )
    for key, order in reading_order_dict.items():
        region = bs_data.find(attrs={'id': f'{id_list[int(key)]}'})
        custom_match = re.search(
            r"(structure \{type:.+?;})", region["custom"]
        )

        class_info = "structure {type:UnkownRegion;}" if custom_match is None else custom_match.group(1)
        region.attrs['custom'] = f"readingOrder {{index:{order};}} {class_info}"

        # TODO: align this with convert xml into shared functions
        order_group.append(
            bs_copy.new_tag(
                "RegionRefIndexed",
                attrs={"index": str(order), "regionRef": id_list[int(key)]},
            )
        )
        page.append(region)


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
