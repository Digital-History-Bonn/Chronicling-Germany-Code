"""Mdule for plotting worker experiment"""
import json
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
# import tikzplotlib
from numpy import ndarray
from tqdm import tqdm


# def tikzplotlib_fix_ncols(obj):
#     """
#     workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
#     """
#     if hasattr(obj, "_ncols"):
#         obj._ncol = obj._ncols
#     for child in obj.get_children():
#         tikzplotlib_fix_ncols(child)


def load_json(path: str, shape: Tuple[int, ...], num_batches: int) -> ndarray:
    """
    Load all json files containing data for one training each.
    :param path: path to folder
    :param shape: worker range x prefetch range
    :param num_batches: total number of batches that have been processed during training
    """
    file_names = [f for f in os.listdir(path) if f.endswith(".json")]
    data = np.zeros(shape)
    for file in tqdm(file_names, desc="loading data", unit="files"):
        with open(f"{path}{file}", "r", encoding="utf-8") as file:
            worker, prefetch, duration = json.load(file)
            data[worker - 1][prefetch - 1] = num_batches / duration
    return data


def plot(data: ndarray) -> None:
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

    # fig = plt.gcf()
    # fig = tikzplotlib_fix_ncols(fig)
    # tikzplotlib.save('test.tex')

    plt.savefig('test.pdf')

    plt.show()


data_array = load_json("logs/worker-data/", (40, 10), 79 * 5)
print(np.argmax(data_array))
plot(data_array)
