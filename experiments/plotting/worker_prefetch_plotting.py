"""Module for plotting worker and prefetch factor graphs"""
import json
import os
from typing import Tuple, Any, List

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
            worker, prefetch, duration = json.load(file)
            data[worker - 1][prefetch - 1] = num_batches / duration
    return data


# def load_json(path: str) -> List[List[float]]:
#     """
#     Load all json files containing data for one training each.
#     :param path: path to folder
#     :param shape: intended shape of resulting data
#     :param num_batches: total number of batches that have been processed during training
#     """
#     worker = 79 * 3
#     file_names = [f for f in os.listdir(path) if f.endswith(".json")]
#     values = []
#     for file_name in tqdm(file_names, desc="loading data", unit="files"):
#         with open(f"{path}{file_name}", "r", encoding="utf-8") as file:
#             raw_data = json.load(file)
#             values.append([raw_data[1], worker/raw_data[4]])
#     return values


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

    axplt.set_title("Batch Verarbeitung")
    axplt.set_xlabel('Prefetch Faktor')
    axplt.set_ylabel('Anzahl Worker')
    axplt.set_zlabel('Batches pro Sekunde')
    axplt.set_xticks(np.arange(6) * 2 - 0.6, np.arange(6) * 2)
    axplt.set_yticks(np.arange(5) * 10 - 2, np.arange(5) * 10)
    axplt.view_init(elev=20., azim=135)

    # fig = plt.gcf()
    # fig = tikzplotlib_fix_ncols(fig)
    # tikzplotlib.save('test.tex')

    plt.savefig('worker-3d.pdf')

    plt.show()


def plot_2d(data: ndarray, stds: ndarray, name: str, xlabel: str, ticks: Any) -> None:
    """
    Plot 2d data, which has been summarized from 3d Data along one axis.
    :param data: ndarray 2d data
    :param stds: standart deviation ndarray fro each data point
    :param name: name for saved file
    :param xlabel: name for x-axis label
    """
    fig, axplt = plt.subplots()
    axplt.set_xlabel(xlabel)
    axplt.set_ylabel('Batches pro Sekunde')
    axplt.bar(data[:, 0] - 1, data[:, 1], align="center", yerr=stds[:, 1])
    axplt.set_xticks(ticks[0], ticks[1])

    plt.savefig(f"{name}.pdf")

    # pylint: disable=assignment-from-no-return
    fig = plt.gcf()
    fig = tikzplotlib_fix_ncols(fig)
    tikzplotlib.save(f"{name}.tex")
    plt.show()


data_ndarray = load_json("logs/worker-data-11-11/", (40, 10), 79 * 5)
print(np.argmax(data_ndarray))
plot_3d(data_ndarray)
data_2d = np.mean(data_ndarray, axis=0)
std_ndarray = np.std(data_ndarray, axis=0)
plot_2d(data_2d, std_ndarray, "prefetch-2d", "Prefetch Faktor", (np.arange(6) * 2 - 1, np.arange(6) * 2))

data_2d = np.mean(data_ndarray, axis=1)
std_ndarray = np.std(data_ndarray, axis=1)
plot_2d(data_2d, std_ndarray, "worker-2d", "Anzahl Worker", (np.arange(5) * 10 - 1, np.arange(5) * 10))

# values_1 = np.roll(np.array(load_json("logs/cbam_worker/")), 0, axis=0)
# values_2 = np.roll(np.array(load_json("logs/cbam_worker_B/")), 0, axis=0)
# values_3 = np.roll(np.array(load_json("logs/cbam_worker_C/")), 0, axis=0)
#
# data = np.stack([values_1, values_2, values_3])
# mean_data = np.mean(data, axis=0)
# error = np.std(data, axis=0)
#
# plot_2d(mean_data, error, "cbam-worker-2d", "Anzahl Worker", (np.arange(9) * 5 - 1, np.arange(9) * 5))
