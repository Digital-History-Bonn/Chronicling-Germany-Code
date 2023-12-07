"""Module for plotting hyperparameter search results"""
import json
import os
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib
from numpy import ndarray
from tqdm import tqdm
from experiments.plotting.utils import tikzplotlib_fix_ncols


# pylint: disable=duplicate-code
def load_json(path: str) -> List[List[float]]:
    """
    Load all json files containing data for one training each.
    :param path: path to folder
    :param shape: intended shape of resulting data
    :param num_batches: total number of batches that have been processed during training
    """
    file_names = [f for f in os.listdir(path) if f.endswith(".json")]
    values = []
    for file_name in tqdm(file_names, desc="loading data", unit="files"):
        with open(f"{path}{file_name}", "r", encoding="utf-8") as file:
            values.append(json.load(file))
    return values


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


def main():
    values_1 = np.roll(np.array(load_json("logs/batch_A/")), 0, axis=0)
    values_2 = np.roll(np.array(load_json("logs/batch_B/")), 0, axis=0)
    values_3 = np.roll(np.array(load_json("logs/batch_C/")), 0, axis=0)

    data = np.stack([values_1, values_2, values_3])
    mean_data = np.mean(data, axis=0)
    error = np.std(data, axis=0)

    # print(np.argmax(data_ndarray))
    plot_2d(mean_data, error, "batch-hyper")


if __name__ == "__main__":
    main()
