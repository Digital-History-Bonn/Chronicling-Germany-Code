"""Module for plotting loss function and amp 2d graph"""
import json
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib
from numpy import ndarray
from tqdm import tqdm
from experiments.plotting.utils import tikzplotlib_fix_ncols


# pylint: disable=duplicate-code
def load_json(path: str, shape: Tuple[int, ...], num_batches: int) -> ndarray:
    """
    Load all json files containing data for one training each.
    :param path: path to folder
    :param shape: intended shape of resulting data
    :param num_batches: total number of batches that have been processed during training
    """
    file_names = [f for f in os.listdir(path) if f.endswith(".json")]
    data = np.zeros(shape)
    for file_name in tqdm(file_names, desc="loading data", unit="files"):
        with open(f"{path}{file_name}", "r", encoding="utf-8") as file:
            loss, amp, duration = json.load(file)
            index = 0 if loss == "cross_entropy" else 1
            letter = file_name[-6]
            iteration = 0 if letter == "A" else 1 if letter == "B" else 2
            data[index][int(amp)][iteration] = num_batches / duration
    return data


def plot_2d(data: ndarray, stds: ndarray, name: str) -> None:
    """
    Plot 2d data, which has been summarized from 3d Data along one axis.
    :param data: ndarray 2d data
    :param stds: standart deviation ndarray fro each data point
    :param name: name for saved file
    :param xlabel: name for x-axis label
    """
    labels = ["no amp", "amp"]
    xdata = np.arange(data.shape[0]) - 0.125
    fig, axplt = plt.subplots()
    axplt.set_ylabel('Batches pro Sekunde')
    for i in range(2):
        axplt.bar(xdata + i / 4, data[:, i], 0.25, align="center", yerr=stds[:, i], label=labels[i])

    axplt.legend(loc='upper right')
    axplt.set_xticks([0, 1], ["cross entropy", "focal loss"])
    plt.savefig(f"{name}.pdf")
    fig = plt.gcf()
    # pylint: disable=assignment-from-no-return
    fig = tikzplotlib_fix_ncols(fig)
    tikzplotlib.save(f"{name}.tex")
    plt.show()


data_ndarray = load_json("logs/amp-loss-data-11-11/", (2, 2, 3), 79 * 5)
print(np.argmax(data_ndarray))
data_2d = np.mean(data_ndarray, axis=2)
std_ndarray  = np.std(data_ndarray, axis=2)
plot_2d(data_2d, std_ndarray, "loss-amp")
