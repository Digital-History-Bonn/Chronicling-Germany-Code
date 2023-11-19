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


def plot_2d(data: ndarray, name: str) -> None:
    """
    Plot 2d data, which has been summarized from 3d Data along one axis.
    :param values: ndarray containing Lists of 4 values: wd, lr, score, class score
    :param name: name for saved file
    """
    labels = ["score", "class score"]
    main_color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    background_color = ['lightsteelblue', 'peachpuff', 'palegreen', 'tab:red', 'tab:purple']

    plt.plot(data[:,0], data[:,2], color=main_color[0], label=labels[0])
    plt.plot(data[:,0], data[:,3], color=main_color[1], label=labels[1])
    # plt.fill_between(steps[index], mean - error, mean + error, color=background_color[index])
    plt.title("Weight Decay")

    plt.xscale("log")

    plt.xlabel("Weight Decay")
    plt.legend(loc="upper right")

    plt.savefig(f"{name}.pdf")
    fig = plt.gcf()
    fig = tikzplotlib_fix_ncols(fig)
    tikzplotlib.clean_figure()
    tikzplotlib.save(f"{name}.tex")


def main():
    values = load_json("logs/learning-rate-test/")
    # print(np.argmax(data_ndarray))
    plot_2d(np.roll(np.array(values), 1, axis=0), "wd-hyper")


if __name__ == "__main__":
    main()
