import numpy as np
import tikzplotlib
from matplotlib import pyplot as plt
from numpy import ndarray

from experiments.plotting.utils import tikzplotlib_fix_ncols


def plot_2d(data: ndarray, stds: ndarray, name: str) -> None:
    """
    Plot 2d data, which has been summarized from 3d Data along one axis.
    :param data: ndarray 2d data
    :param stds: standart deviation ndarray fro each data point
    :param name: name for saved file
    :param xlabel: name for x-axis label
    """
    labels = ["ub mannheim", "bonn with pretraining"]
    xdata = np.arange(data.shape[0]) - 0.125
    fig, axplt = plt.subplots()
    axplt.set_ylabel("")
    for i in range(2):
        axplt.bar(xdata + i / 4, data[:, i], 0.25, align="center", label=labels[i])

    axplt.legend(loc='upper left')
    axplt.set_xticks([0, 1], ["distance per 1000 characters", "correct line percentage"])
    plt.savefig(f"{name}.pdf")
    fig = plt.gcf()
    # pylint: disable=assignment-from-no-return
    fig = tikzplotlib_fix_ncols(fig)
    tikzplotlib.save(f"{name}.tex")
    plt.show()


def main():
    values_1 = np.array([0.15030, 0.10866])  # distance/1000 characters
    values_2 = np.array([0.47846, 0.67592])  # correct

    data = np.stack([values_1, values_2])
    # mean_data = np.mean(data, axis=0)
    # error = np.std(data, axis=0)

    # print(np.argmax(data_ndarray))
    plot_2d(data, np.array([]), "ocr_results")


if __name__ == "__main__":
    main()
