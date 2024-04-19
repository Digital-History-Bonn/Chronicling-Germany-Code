from typing import List, Dict, Tuple, Union

import numpy as np
from numpy import ndarray

from transkribus_export import bbox_sufficient


def get_region_properties(bbox_dict: Dict[int, List[List[float]]]) -> ndarray:
    """

    Takes Dict with label keys and bbox List and converts it to a property ndarray.
    :param bbox_dict: Label keys and bbox Lists with minx, miny, maxx, maxy
    :return: 2d ndarray with n x 6 values. Containing id, label, minx, maxx, meanx and meany, width, height.
    """
    index = 0
    result = []
    for label, bbox_list in bbox_dict.items():
        if len(bbox_list) == 0:
            continue
        bbox_ndarray = np.array(bbox_list)
        label_properties = np.vstack(
            [np.arange(len(bbox_ndarray)) + index, np.full(len(bbox_ndarray), label), bbox_ndarray[:, 0],
             bbox_ndarray[:, 2], np.mean([bbox_ndarray[:, 0], bbox_ndarray[:, 2]], axis=0),
             np.mean([bbox_ndarray[:, 1], bbox_ndarray[:, 3]], axis=0), bbox_ndarray[:, 2] - bbox_ndarray[:, 0],
             bbox_ndarray[:, 3] - bbox_ndarray[:, 1]]).T
        result.append(label_properties)
        index += len(bbox_ndarray)
    return np.concatenate(result, axis=0)


def median(properties: ndarray) -> int:
    """
    Calculates the mean of x width of all text and table regions. This produces an estimation for column size.
    :param properties: 2d ndarray with n x 6 values. Containing id, label, minx, maxx, meanx, meany, width
        and height.
    :return: text region x mean
    """
    labels = properties[:, 1]
    text_regions = properties[(labels == 3) + (labels == 4)]
    return np.median(text_regions[:, 3] - text_regions[:, 2])  # type: ignore


def get_global_splitting_regions(properties: ndarray, x_median: int, columns_per_page: int) -> ndarray:
    """
    Determine divider indices. A region is considered a page divider, if it has the right class and if it
    spans over the whole page. This means, it has to be at least as wide, as x_median * columns_per_page.
    :param properties: 2d ndarray with n x 6 values. Containing id, label, minx, maxx, meanx, meany, width
        and height.
    :param x_median: estimated column width.
    :param columns_per_page: estimated columns per page.
    :return: index list of page spanning section divider
    """
    class_indices, rounded_width = get_width_in_terms_of_column(properties, x_median, 9)
    splitting_regions = np.zeros(properties.shape[0], dtype=bool)
    splitting_regions[class_indices] = rounded_width >= columns_per_page
    return splitting_regions  # type: ignore


def get_local_splitting_regions(properties: ndarray, x_median: int, columns_per_page: int) -> ndarray:
    """
    Determine divider indices. A region is considered a page divider, if it has the right class and if it
    spans over the whole page. This means, it has to be at least as wide, as x_median * columns_per_page.
    :param properties: 2d ndarray with n x 6 values. Containing id, label, minx, maxx, meanx, meany, width
        and height.
    :param x_median: estimated column width.
    :param columns_per_page: estimated columns per page.
    :return: index list of page spanning section divider
    """
    class_indices, rounded_width = get_width_in_terms_of_column(properties, x_median, 9)
    splitting_regions = np.zeros(properties.shape[0], dtype=bool)
    splitting_regions[class_indices] = (rounded_width < columns_per_page) * (rounded_width > 0)
    return splitting_regions  # type: ignore


def get_width_in_terms_of_column(properties: ndarray, x_median: int, region_class: int) -> Tuple[ndarray, ndarray]:
    """
    Calculates the width of all regions with specified class in terms of columns. For example a region can be roughly
    2 columns wide.
    :param properties: 2d ndarray with n x 6 values. Containing id, label, minx, maxx, meanx, meany, width
        and height.
    :param x_median: estimated column width.
    :param region_class: class number to be filtered by.
    :return: index and rounded width lists
    """
    class_indices: ndarray = np.where(properties[:, 1] == region_class)[0]
    rounded_width: ndarray = np.round((properties[class_indices][:, 3] - properties[class_indices][:, 2]) / x_median)
    return class_indices, rounded_width


def sort_column(column: ndarray) -> Tuple[ndarray, ...]:
    """
    Splits column regions into left and right border separators, which separatre columns from one another if they
    are present. Those are sorted from left to right. All other column regions are sorted from top to bottom.
    :param columns: all regions in columns
    :param column: all regions belonging to this column.
    :return: sorted ndarrays for left and right border as well as the actual column regions.
    """
    is_separator = np.where(np.isin(column[:, 1], [7, 8, 9]))[0]
    column_x_mean = np.mean(column[np.invert(is_separator)][:, 4])
    is_vertical = (
            column[is_separator][:, 6] < column[is_separator][:, 7])

    is_right_of_column = column[is_separator][:, 2] > column_x_mean
    is_left_of_column = column[is_separator][:, 3] < column_x_mean

    is_right_border_separator = is_right_of_column * is_vertical
    is_left_border_separator = is_left_of_column * is_vertical
    is_border_separator = is_separator[is_right_border_separator + is_left_border_separator]

    column_to_sort = np.delete(column, is_border_separator, axis=0)
    left_border_to_sort = column[is_separator][is_left_border_separator]
    right_border_to_sort = column[is_separator][is_right_border_separator]

    sorted_column = column_to_sort[np.argsort(column_to_sort[:, 5])]
    sorted_left_border_separator = left_border_to_sort[np.argsort(left_border_to_sort[:, 4])]
    sorted_right_border_separator = right_border_to_sort[np.argsort(right_border_to_sort[:, 4])]

    return sorted_column, sorted_left_border_separator, sorted_right_border_separator


class PageProperties:
    def __init__(self, bbox_dict: Dict[int, List[List[float]]], properties: Union[None, ndarray] = None):
        if properties is not None:
            assert properties.shape[1] == 8, f"Properties ndarray has to have the shape nx8. Found {properties.shape}"
            self.properties = properties
        else:
            self.properties = get_region_properties(bbox_dict)

        self.x_median = median(self.properties)
        self.global_x_min = np.min(self.properties[:, 2])
        self.global_x_max = np.max(self.properties[:, 3])
        self.columns_per_page: int = round((self.global_x_max - self.global_x_min) / self.x_median)

        self.global_splitting_bool = get_global_splitting_regions(self.properties, self.x_median,
                                                                  self.columns_per_page)
        self.local_splitting_bool = get_local_splitting_regions(self.properties, self.x_median,
                                                                self.columns_per_page)

    def get_reading_order(self) -> Dict[int, int]:
        """Calculates reading order by splitting the page into sections and columns. All regions in one column
        are then sorted vertically."""
        result: List[int] = []
        self.global_divider_split(self.properties, self.global_splitting_bool, self.local_splitting_bool, result)

        return {k: v for v, k in enumerate(result)} # TODO: why dict? just preserve list

    def global_divider_split(self, properties: ndarray, global_splitting_bool: ndarray,
                             local_splitting_indices: ndarray,
                             result: List[int]) -> None:
        """
        If there are still global dividers in the properties sub array, a global split is executed.
        This includes, recursively recalling this function.
        :param properties: 2d ndarray with n x 6 values. Containing id, label, minx, maxx, meanx, meany, width
        and height.
        :param global_splitting_bool: Indices of page spanning dividers.
        :param local_splitting_indices: Indices for dividers that span over at least 2 columns, but not over the whole page.
        :param result: reading order result which is updated by reference
        """
        if any(global_splitting_bool):
            self.execute_single_global_split(properties, global_splitting_bool, local_splitting_indices, result)
        else:
            self.local_divider_split(properties, local_splitting_indices, result)

    def execute_single_global_split(self, properties: ndarray, global_splitting_bool: ndarray,
                                    local_splitting_bool: ndarray, result: List[int]) -> None:
        """
        Executes global divider split by deleting the divider and calling global_divider_split for all regions above
        and below the divider separately. The divider itself is inserted between the regions above and below in the
        reading order result.
        :param global_splitting_bool: Indices of page spanning dividers.
        :param local_splitting_bool: Indices for dividers that span over at least 2 columns, but not over the whole page.
        :param properties: 2d ndarray with n x 6 values. Containing id, label, minx, maxx, meanx, meany, width
        and height.
        :param result: reading order result which is updated by reference
        """
        split_index = np.where(global_splitting_bool)[0][np.argmax(properties[global_splitting_bool][:, 5])]
        divider_entry = properties[split_index]

        properties = np.delete(properties, split_index, axis=0)
        global_splitting_bool = np.delete(global_splitting_bool, split_index, axis=0)
        local_splitting_bool = np.delete(local_splitting_bool, split_index, axis=0)

        region_bool = properties[:, 5] > divider_entry[5]

        self.global_divider_split(properties[np.invert(region_bool)], global_splitting_bool[np.invert(region_bool)],
                                  local_splitting_bool[np.invert(region_bool)], result)
        result.append(divider_entry[0])  # type: ignore
        self.global_divider_split(properties[region_bool], global_splitting_bool[region_bool],
                                  local_splitting_bool[region_bool], result)

    def local_divider_split(self, properties: ndarray, local_splitting_bool: ndarray, result: List[int]) -> None:
        """
        If there are still local dividers in the properties sub array, a global split is executed.
        This includes, recursively recalling this function.
        :param properties: 2d ndarray with n x 6 values. Containing id, label, minx, maxx, meanx, meany, width
        and height.
        :param local_splitting_bool: Indices for dividers that span over at least 2 columns, but not over the whole page.
        :param result: reading order result which is updated by reference
        """
        if any(local_splitting_bool):
            self.execute_single_local_split(properties, local_splitting_bool, result)
        else:
            self.column_split(properties, result)

    def execute_single_local_split(self, properties: ndarray, local_splitting_bool: ndarray,
                                   result: List[int]) -> None:
        """
        Executes local divider split by deleting the divider and calling local_divider_split for all regions that
        are above, left or right of the divider. For the remaining regions below the devider local_divider_split is called
        again. The divider itself is inserted bevore the remaining regions into the reading order result.
        :param global_splitting_indices: Indices of page spanning dividers.
        :param local_splitting_bool: Indices for dividers that span over at least 2 columns, but not over the whole page.
        :param properties: 2d ndarray with n x 6 values. Containing id, label, minx, maxx, meanx, meany, width
        and height.
        :param result: reading order result which is updated by reference
        """
        split_index = np.where(local_splitting_bool)[0][np.argmin(properties[local_splitting_bool][:, 5])]
        divider_entry = properties[split_index]

        properties = np.delete(properties, split_index, axis=0)
        local_splitting_bool = np.delete(local_splitting_bool, split_index, axis=0)

        region_bool = (properties[:, 5] < divider_entry[5]) + (properties[:, 4] > divider_entry[3]) + (
                properties[:, 4] < divider_entry[2])

        self.local_divider_split(properties[region_bool], local_splitting_bool[region_bool], result)
        result.append(divider_entry[0])  # type: ignore
        self.local_divider_split(properties[np.invert(region_bool)], local_splitting_bool[np.invert(region_bool)],
                                 result)

    def column_split(self, properties: ndarray, result: List[int]) -> None:
        """
        Estimates the right column border by taking the global x min value and adding it to the x median. From there,
        all regions left of the estimated column border are considered to be in the column and are sorted.
        The dorted indices are appended to the result list. Then, the next column border is
        estimated and the already processed column regions are excluded from further analysis. If no region left of
        the column border is found, the loop continues. If all regions that are left, are in that column, the
        loop terminates.
        :param properties: 2d ndarray with n x 6 values. Containing id, label, minx, maxx, meanx, meany, width
        and height.
        :param result: reading order result which is updated by reference
        """
        if len(properties) == 0:
            return

        estimated_column_border = self.global_x_min + self.x_median
        columns = np.copy(properties)

        while True:
            in_column: ndarray = columns[:, 4] < estimated_column_border

            if all(np.invert(in_column)):
                estimated_column_border += self.x_median
                continue

            sorted_column, sorted_left_border_separator, sorted_right_border_separator = sort_column(columns[in_column])
            result += sorted_left_border_separator[:, 0].tolist()
            result += sorted_column[:, 0].tolist()
            result += sorted_right_border_separator[:, 0].tolist()

            if all(in_column):
                break

            actual_column_border = np.mean(sorted_column[np.isin(sorted_column[:, 1], [2, 3, 4, 5])][:, 3])
            estimated_column_border = actual_column_border + self.x_median if not np.isnan(actual_column_border) else estimated_column_border + self.x_median
            columns = columns[np.invert(in_column)]
