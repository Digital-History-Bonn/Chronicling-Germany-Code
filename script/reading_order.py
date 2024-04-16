from typing import List, Dict, Tuple

import numpy as np
from numpy import ndarray

from transkribus_export import bbox_sufficient


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
    text_regions = properties[(labels == 3) + (labels == 4)]
    return np.median(text_regions[:, 3] - text_regions[:, 2]) # type: ignore


def get_global_splitting_regions(properties: ndarray, x_median: int, columns_per_page: int) -> ndarray:
    """
    Determine divider indices. A region is considered a page divider, if it has the right class and if it
    spans over the whole page. This means, it has to be at least as wide, as x_median * columns_per_page.
    :param properties: 2d ndarray with n x 6 values. Containing id, label, minx, maxx, meanx and meany.
    :param x_median: estimated column width.
    :param columns_per_page: estimated columns per page.
    :return: index list of page spanning section divider
    """
    class_indices, rounded_width = get_width_in_terms_of_column(properties, x_median, 9)
    return class_indices[rounded_width >= columns_per_page]  # type: ignore


def get_local_splitting_regions(properties: ndarray, x_median: int, columns_per_page: int) -> ndarray:
    """
    Determine divider indices. A region is considered a page divider, if it has the right class and if it
    spans over the whole page. This means, it has to be at least as wide, as x_median * columns_per_page.
    :param properties: 2d ndarray with n x 6 values. Containing id, label, minx, maxx, meanx and meany.
    :param x_median: estimated column width.
    :param columns_per_page: estimated columns per page.
    :return: index list of page spanning section divider
    """
    class_indices, rounded_width = get_width_in_terms_of_column(properties, x_median, 9)
    return class_indices[(rounded_width < columns_per_page) * (rounded_width > 0)]  # type: ignore


def get_width_in_terms_of_column(properties: ndarray, x_median: int, region_class: int) -> Tuple[ndarray, ndarray]:
    """
    Calculates the width of all regions with specified class in terms of columns. For example a region can be roughly
    2 columns wide.
    :param properties: 2d ndarray with n x 6 values. Containing id, label, minx, maxx, meanx and meany.
    :param x_median: estimated column width.
    :param class: class number to be filtered by.
    :return: index and rounded width lists
    """
    class_indices = np.where(properties[:, 1] == region_class)[0]
    rounded_width = np.round((properties[class_indices][:, 3] - properties[class_indices][:, 2]) / x_median)
    return class_indices, rounded_width


class PageProperties:
    def __init__(self, bbox_dict: Dict[int, List[List[float]]]):
        self.properties = get_region_properties(bbox_dict)
        self.x_median = median(self.properties)
        self.global_x_min = np.min(self.properties[:, 2])
        self.global_x_max = np.max(self.properties[:, 3])
        self.columns_per_page: int = round(self.global_x_max - self.global_x_min / self.x_median)

        self.global_splitting_indices = get_global_splitting_regions(self.properties, self.x_median,
                                                                     self.columns_per_page)
        self.local_splitting_indices = get_local_splitting_regions(self.properties, self.x_median,
                                                                   self.columns_per_page)

    def get_reading_order(self) -> Dict[int, int]:
        """"""
        result: List[int] = []
        self.global_divider_split(self.properties, self.global_splitting_indices, self.local_splitting_indices, result)

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

    def global_divider_split(self, properties: ndarray, global_splitting_indices: ndarray,
                             local_splitting_indices: ndarray,
                             result: List[int]) -> None:
        """
        If there are still global dividers in the properties sub array, a global split is executed.
        This includes, recursively recalling this function.
        :param properties: 2d ndarray with n x 6 values. Containing id, label, minx, maxx, meanx and meany.
        :param global_splitting_indices: Indices of page spanning dividers.
        :param local_splitting_indices: Indices for dividers that span over at least 2 columns, but not over the whole page.
        :param result: reading order result which is updated by reference
        """
        if len(global_splitting_indices) > 0:
            self.execute_single_global_split(properties, global_splitting_indices, local_splitting_indices, result)
        else:
            self.local_divider_split(properties, local_splitting_indices, result)

    def execute_single_global_split(self, properties: ndarray, global_splitting_indices: ndarray,
                                    local_splitting_indices: ndarray, result: List[int]) -> None:
        """
        Executes global divider split by deleting the divider and calling global_divider_split for all regions above
        and below the divider separately. The divider itself is inserted between the regions above and below in the
        reading order result.
        :param global_splitting_indices: Indices of page spanning dividers.
        :param local_splitting_indices: Indices for dividers that span over at least 2 columns, but not over the whole page.
        :param properties: 2d ndarray with n x 6 values. Containing id, label, minx, maxx, meanx and meany.
        :param result: reading order result which is updated by reference
        """
        index = global_splitting_indices[0]
        divider_entry = properties[index]

        properties = np.delete(properties, index, axis=0)
        global_splitting_indices = np.delete(global_splitting_indices, 0, axis=0)

        region_bool = properties[:, 5] > divider_entry[5]
        global_bool = region_bool[global_splitting_indices]
        local_bool = region_bool[local_splitting_indices]

        self.global_divider_split(properties[np.invert(region_bool)], global_splitting_indices[np.invert(global_bool)],
                                  local_splitting_indices[np.invert(local_bool)], result)
        result.append(divider_entry[0])  # type: ignore
        self.global_divider_split(properties[region_bool], global_splitting_indices[global_bool],
                                  local_splitting_indices[local_bool], result)

    def local_divider_split(self, properties: ndarray, local_splitting_indices: ndarray, result: List[int]) -> None:
        """
        If there are still local dividers in the properties sub array, a global split is executed.
        This includes, recursively recalling this function.
        :param properties: 2d ndarray with n x 6 values. Containing id, label, minx, maxx, meanx and meany.
        :param local_splitting_indices: Indices for dividers that span over at least 2 columns, but not over the whole page.
        :param result: reading order result which is updated by reference
        """
        if len(local_splitting_indices) > 0:
            self.execute_single_local_split(properties, local_splitting_indices, result)
        else:
            pass

    def execute_single_local_split(self, properties: ndarray, local_splitting_indices: ndarray,
                                   result: List[int]) -> None:
        """
        Executes local divider split by deleting the divider and calling local_divider_split for all regions that
        are above, left or right of the divider. For the remaining regions below the devider local_divider_split is called
        again. The divider itself is inserted bevore the remaining regions into the reading order result.
        :param global_splitting_indices: Indices of page spanning dividers.
        :param local_splitting_indices: Indices for dividers that span over at least 2 columns, but not over the whole page.
        :param properties: 2d ndarray with n x 6 values. Containing id, label, minx, maxx, meanx and meany.
        :param result: reading order result which is updated by reference
        """
        index = local_splitting_indices[0]
        divider_entry = properties[index]

        properties = np.delete(properties, index, axis=0)
        local_splitting_indices = np.delete(local_splitting_indices, 0, axis=0)

        region_bool = (properties[:, 5] > divider_entry[5]) + (properties[:, 4] > divider_entry[3]) + (
                properties[:, 4] < divider_entry[2])
        local_bool = region_bool[local_splitting_indices]

        self.local_divider_split(properties[region_bool], local_splitting_indices[local_bool], result)
        result.append(divider_entry[0])  # type: ignore
        self.local_divider_split(properties[np.invert(region_bool)], local_splitting_indices[np.invert(local_bool)],
                                 result)

    def column_split(self, properties: ndarray, result: List[int]) -> None:
        estimated_column_border = self.global_x_min + self.x_median

        separator_indices = np.where((properties[:, 1] == 8) or (properties[:, 1] == 7))[0]
        np.where(np.abs(properties[:, 4] - estimated_column_border))

        region_bool = properties[:, 5] > estimated_column_border
