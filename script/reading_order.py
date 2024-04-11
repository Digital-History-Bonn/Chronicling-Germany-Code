from typing import List, Dict

import numpy as np
from numpy import ndarray

from transkribus_export import bbox_sufficient


# def get_splitting_regions(bbox_list: ndarray, big_separator_size:int) -> List[int]:
#     """
#     Extract big separators if they meet the threshold
#     :param bbox_list: 2d n x 6 ndarray with id, label and bbox corners.
#     :param big_separator_size: big separator threshold. If this requirement isnt met, the separator is too short
#     :return: list of big separators
#     """
#     index_list = np.where(bbox_list[:, 1] == 9)[0]
#     result_list = []
#     for index in index_list:
#         if bbox_sufficient(bbox_list[index][2:], big_separator_size, True):
#             result_list.append(index)
#     return result_list
#
#
# def get_reading_order(bbox_list: ndarray, result: List[int], big_separator_size: int) -> None:
#     """
#     Calculate reading order by first seperating regions by big seperators. Regions without big seperators are
#     forwarded to calculate_reading_order. Big seperators are being handelt seperately.
#     :param bbox_list: 2d n x 6 ndarray with id, label and bbox corners.
#     :param result: Result List, is being filled over recursive calls.
#     :return: list of indices in reading order
#     """
#
#     splitting_indices= get_splitting_regions(bbox_list, big_separator_size)
#     if len(splitting_indices) > 0:
#         splitting_index = splitting_indices[0]
#         big_seperator_entry = bbox_list[splitting_index]
#         bbox_list = np.delete(bbox_list, splitting_index, axis=0)
#
#         region_bool = bbox_list[:, 5] > big_seperator_entry[3]
#         get_reading_order(bbox_list[np.invert(region_bool)], result, big_separator_size)
#         result.append(big_seperator_entry[0])
#
#         get_reading_order(bbox_list[region_bool], result, big_separator_size)
#     else:
#         calculate_reading_order(bbox_list, result)
#
#
# def calculate_reading_order(bbox_list: ndarray, result: List[int]) -> None:
#     """
#     Receives regions without big sperators.
#     Bboxes are sorted by the sum of the upper left corner to identify the upper left most element.
#     Then, all elements, which begin below of that pivot element are considered one column and sorted verticly.
#     This is repeated until all regions are concatenated.
#     :param bbox_list:
#     :param result:
#     """
#     if bbox_list.size == 0:
#         return
#     sorted_by_sum = bbox_list[np.argsort(bbox_list[:, 2: 4].sum(axis=1))]
#     while True:
#         level_bool = sorted_by_sum[:, 2] <= sorted_by_sum[0, 4]
#         # debug pivot elments
#         # print(f"Pivot Element {len(result) + 1} with bbox {sorted_by_sum[0]}")
#         current_level = sorted_by_sum[level_bool]
#         current_level = current_level[np.argsort(current_level[:, 3])]
#
#         result += list(current_level[:, 0])
#
#         next_level = sorted_by_sum[np.invert(level_bool)]
#         sorted_by_sum = next_level
#         if next_level.size == 0:
#             break
#
#
# def setup_reading_order(args, bbox_list):
#     """"""
#     bbox_ndarray = create_bbox_ndarray(bbox_list)
#     reading_order: List[int] = []
#     get_reading_order(bbox_ndarray, reading_order, int(args.separator_size * args.scale))
#     reading_order_dict = {k: v for v, k in enumerate(reading_order)}
#     return reading_order_dict
#
#
# def create_bbox_ndarray(bbox_dict: Dict[int, List[List[float]]]) -> ndarray:
#     """
#     Takes Dict with label keys and bbox List and converts it to bbox ndarray.
#     :param bbox_dict: Label keys and bbox Lists
#     :return: 2d ndarray with n x 7 values. Containing id, label, 2 bbox corners and x-axis center.
#     """
#     index = 0
#     result = []
#     for label, bbox_list in bbox_dict.items():
#         for bbox in bbox_list:
#             result.append([index, label] + bbox)
#             index += 1
#     return np.array(result, dtype=int)

def get_region_properties(bbox_dict: Dict[int, List[List[float]]]) -> ndarray:
    """
    minx, miny, maxx, maxy
    Takes Dict with label keys and bbox List and converts it to a property ndarray.
    :param bbox_dict: Label keys and bbox Lists
    :return: 2d ndarray with n x 6 values. Containing id, label, minx, maxx, meanx and meany.
    """
    index = 0
    result = []
    for label, bbox_list in bbox_dict.items():
        for bbox in bbox_list:
            result.append([index, label, bbox[0], bbox[2], np.mean([bbox[0], bbox[2]]), np.mean([bbox[1], bbox[3]])])
            index += 1
    return np.array(result, dtype=int)


def median(properties: ndarray) -> int:
    """
    Calculates the mean of x width of all text and table regions. This produces an estimation for column size.
    :param properties: 2d ndarray with n x 6 values. Containing id, label, minx, maxx, meanx and meany.
    :return: text region x mean
    """
    labels = properties[:, 1]
    text_regions = np.where([labels == 3, labels == 4], properties, properties)[0]
    return np.median(text_regions[:, 3] - text_regions[:, 2])


def get_reading_order(bbox_dict: Dict[int, List[List[float]]]) -> Dict[int, int]:
    """"""
    properties = get_region_properties(bbox_dict)
    x_median = median(properties)
    global_x_min = np.min(properties[:, 2])
    global_x_max = np.max(properties[:, 3])
    columns_per_page: int = round(global_x_max - global_x_min / x_median)
    divider_split(properties, x_median, columns_per_page)


def divider_split(properties: ndarray, x_median: int, columns_per_page: int):

    splitting_indices = get_preliminary_splitting_regions(properties, x_median, columns_per_page)

    if len(splitting_indices) > 0:
        index = splitting_indices[0]
        big_seperator_entry = properties[index]
        bbox_list = np.delete(properties, index, axis=0)

        region_bool = bbox_list[:, 5] > big_seperator_entry[5]
        divider_split(bbox_list[np.invert(region_bool)], result, big_separator_size)
        result.append(big_seperator_entry[0])

        get_reading_order(bbox_list[region_bool], result, big_separator_size)

def get_preliminary_splitting_regions(properties: ndarray, x_median: int, columns_per_page) -> List[int]:
    """
    Extract section dividers if they span over the whole page.
    :param bbox_list: 2d ndarray with n x 6 values. Containing id, label, minx, maxx, meanx and meany.
    :param big_separator_size: big separator threshold. If this requirement isnt met, the separator is too short
    :return: list of page spanning section deviders
    """
    divider_list = np.where(properties[:, 1] == 9, properties, properties)[0]
    divider_indices = np.where(np.round((divider_list[:, 3] - divider_list[:, 2]) / x_median) == columns_per_page)[0]
    return divider_indices.tolist()  # type: ignore
