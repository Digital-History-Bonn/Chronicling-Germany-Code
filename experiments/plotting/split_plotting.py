"""
Module for plotting threads and data splitting graph
"""
import json
import math
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
    for file in tqdm(file_names, desc="loading data", unit="files"):
        with open(f"{path}{file}", "r", encoding="utf-8") as file:
            threads, splits, duration = json.load(file)
            index = int(math.log(threads, 2))
            factor = 1 if math.log(splits, 2) == index else 2
            data[index][factor-1] = num_batches / duration
    return data


def plot_3d(data: ndarray) -> None:
    """
    Plot 2d data into 3d plot
    :param data: 2d ndarray with dimensions workers, prefetch factor
    """
    xdata = np.arange(data.shape[1])
    ydata = np.arange(data.shape[0])

    xmesh, ymesh = np.meshgrid(xdata, ydata)
    xmesh = xmesh.ravel()
    ymesh = ymesh.ravel()
    zmesh = np.zeros_like(xmesh)

    width = depth = 0.3

    fig = plt.figure()

    axplt = fig.add_subplot(projection='3d')
    # axplt.xlabel("Prefetch Faktor")
    # axplt.ylabel("Anzahl Worker")
    # axplt.zlabel("Batches pro Sekunde")
    axplt.bar3d(xmesh, ymesh, zmesh, width, depth, data.ravel(), shade=True, alpha=0.8)

    axplt.set_xlabel('Split Faktor')
    axplt.set_ylabel('Threads')
    axplt.set_zlabel('Batches pro Sekunde')
    axplt.set_xticks([0.2, 1.2], [1, 2])
    axplt.set_yticks(np.arange(6) * 2, 2**(np.arange(6) * 2))
    axplt.view_init(elev=20., azim=135)

    # pylint: disable=assignment-from-no-return
    fig = plt.gcf()
    fig = tikzplotlib_fix_ncols(fig)
    tikzplotlib.clean_figure()
    tikzplotlib.save("threads.tex")

    plt.savefig('threads.pdf')

    plt.show()

# def plot_2d(data: ndarray, stds: ndarray, name: str, xlabel: str) -> None:
#     """
#     Plot 2d data, which has been summarized from 3d Data along one axis.
#     :param data: ndarray 2d data
#     :param stds: standart deviation ndarray fro each data point
#     :param name: name for saved file
#     :param xlabel: name for x-axis label
#     """
#     xdata = np.arange(data.shape[0])
#     fig, axplt = plt.subplots()
#     axplt.set_xlabel(xlabel)
#     axplt.set_ylabel('Batches pro Sekunde')
#     axplt.bar(xdata, data, align="center", yerr=stds)
#
#     plt.savefig(name)
#     plt.show()


def plot_2d(data: ndarray, error, name: str) -> None:
    """
    Plot 2d data, which has been summarized from 3d Data along one axis.
    :param values: ndarray containing Lists of 4 values: wd, lr, score, class score
    :param name: name for saved file
    """
    labels = ["score", "class score"]
    main_color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    background_color = ['lightsteelblue', 'peachpuff', 'palegreen', 'tab:red', 'tab:purple']

    plt.plot(data[:, 0], data[:, 2], color=main_color[0], label=labels[0])
    plt.plot(data[:, 0], data[:, 3], color=main_color[1], label=labels[1])
    plt.fill_between(data[:, 0], data[:, 2] - error[:, 2], data[:, 2] + error[:, 2], color=background_color[0])
    plt.fill_between(data[:, 0], data[:, 3] - error[:, 3], data[:, 3] + error[:, 3], color=background_color[1])
    plt.title("Batch Size")

    single_data = np.array([[0.9563, 0.9573, 0.9579], [0.5807, 0.5671, 0.5863]])
    single_mean = np.mean(single_data, axis=1)
    single_std = np.std(single_data, axis=1)
    plt.errorbar([16], single_mean[0], yerr=single_std[0], fmt='o', color=main_color[0])
    plt.errorbar([16], single_mean[1], yerr=single_std[1], fmt='o', color=main_color[1])

    plt.xticks([16, 32, 64], [16, 32, 64])

    plt.xlabel("Batch Size")
    plt.legend(loc="right")

    plt.grid()

    plt.savefig(f"{name}.pdf")
    fig = plt.gcf()
    fig = tikzplotlib_fix_ncols(fig)
    # tikzplotlib.clean_figure()
    tikzplotlib.save(f"{name}.tex")


data_ndarray = load_json("logs/split-data/", (11, 2), 79 * 5)
print(np.argmax(data_ndarray))
plot_3d(data_ndarray)
