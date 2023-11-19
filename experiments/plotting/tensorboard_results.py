from csv import reader
from typing import Any, Tuple, List
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

import requests
import tikzplotlib

from experiments.plotting.utils import tikzplotlib_fix_ncols

TAGS = ['current+best',
        'epoch',
        'multi-acc-test/class+0',
        'multi-acc-test/class+1',
        'multi-acc-test/class+2',
        'multi-acc-test/class+3',
        'multi-acc-test/class+4',
        'multi-acc-test/class+5',
        'multi-acc-test/class+6',
        'multi-acc-test/class+7',
        'multi-acc-test/class+8',
        'multi-acc-test/class+9',
        'multi-acc-val/class+0',
        'multi-acc-val/class+1',
        'multi-acc-val/class+2',
        'multi-acc-val/class+3',
        'multi-acc-val/class+4',
        'multi-acc-val/class+5',
        'multi-acc-val/class+6',
        'multi-acc-val/class+7',
        'multi-acc-val/class+8',
        'multi-acc-val/class+9',
        'test/accuracy',
        'test/jaccard+score',
        'test/loss',
        'train loss',
        'val/accuracy',
        'val/jaccard+score',
        'val/loss']

# RUNS = [
#     ['cross_entropy_amp_A', 'cross_entropy_amp_B', 'cross_entropy_amp_C'],
#     ['focal_loss_amp_A', 'focal_loss_amp_B', 'focal_loss_amp_C'],
#     ['focal_loss_no_amp_A', 'focal_loss_no_amp_B', 'focal_loss_no_amp_C']
# ]

# RUNS = [
#     ['cross_entropy_amp_A'],
#     ['focal_loss_amp_A'],
#     ['focal_loss_no_amp_A']
# ]

RUNS = [['lerning_rate_test_4_6_A']]

plt.rcParams["figure.figsize"] = (30, 20)
plt.rcParams["font.size"] = 35
plt.rcParams['savefig.pad_inches'] = 0
plt.rcParams.update({'font.size': 40})


def get_data(tag: str, run: str) -> Tuple[ndarray, ndarray]:
    """
    Load tensorboard data from localhost
    :param tag: which data to load from a run
    :param run: which runs to load
    :return: Tuple with steps and data 1d ndarrays
    """
    url = f'http://localhost:6006/experiment/defaultExperimentId/data/plugin/scalars/scalars?tag={tag}&run={run}&format=csv'
    r = requests.get(url, allow_redirects=True)
    data_csv = reader(r.text.splitlines())
    data = np.array(list(data_csv)[1:], dtype=float)
    return data[:, 1], data[:, 2]


def get_timeseries(tag: str, runs: List[List[str]] = RUNS) -> Tuple[List[ndarray], List[ndarray]]:
    """
    Build up lists for each run containing all versions of that run.
    :param tag: tag of data that should be loaded
    :param runs: which runs to use
    :return: steps and data lists
    """
    if runs is None:
        runs = RUNS
    data = []
    step_lists = []
    for run in runs:
        value_list = []
        for version in run:
            steps, values = get_data(tag, version)
            value_list.append(values)
        data.append(np.array(value_list))
        step_lists.append(np.array(steps, dtype=int))

    #custom step correction
    # step_lists[-1] = step_lists[-1]//2
    # step_lists[-2] = step_lists[-2]//2

    return step_lists, data


def average(data):
    avg_data = []
    for i in range(0, len(data), 3):
        avg_data.append(np.mean(data[i:i + 3], axis=0))
    return avg_data


STEPS, EPOCHS = get_timeseries('epoch')


def plot(steps, data, main_color, background_color, title, labels, tiks_name, ylabel, legend):
    """Plots timeseries with error bands"""
    for index, timeseries in enumerate(data):
        # mean = np.mean(timeseries, axis=0)
        # error = np.std(timeseries, axis=0)
        mean = timeseries[0]

        # plt.plot(steps[index], mean, color=main_color[index], label=labels[index])
        plt.plot(steps[index], mean, color=main_color[index])
        # plt.fill_between(steps[index], mean - error, mean + error, color=background_color[index])
    plt.title(title)

    epochs = EPOCHS[0][0].astype(int)
    number_epochs = epochs[-1]
    number_steps = steps[0][-1]
    step_per_epoch = number_steps // number_epochs
    epoch_tiks = 20

    plt.xticks(np.arange(0, number_steps, step_per_epoch * epoch_tiks), np.arange(0, number_epochs + 1, epoch_tiks))

    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    # plt.legend(loc=legend)

    plt.savefig(f"{tiks_name}.pdf")
    fig = plt.gcf()
    fig = tikzplotlib_fix_ncols(fig)
    tikzplotlib.clean_figure()
    tikzplotlib.save(f"{tiks_name}.tex")
    plt.clf()
    # plt.show()


def val_loss():
    steps, data = get_timeseries('val/loss')
    title = "Validation Loss"
    tiks_name = "hyperbest_val_loss"
    ylabel = "Loss"
    legend = "upper right"

    return steps, data, title, tiks_name, ylabel, legend

def val_acc():
    steps, data = get_timeseries('val/accuracy')
    title = "Validation accuracy"
    tiks_name = "hyperbest_val_acc"
    ylabel = "Accuracy"
    legend = "lower right"

    return steps, data, title, tiks_name, ylabel, legend

def val_jac():
    steps, data = get_timeseries('val/jaccard score')
    title = "Validation Jaccard Score"
    tiks_name = "hyperbest_val_jac"
    ylabel = "Jaccard Score"
    legend = "lower right"

    return steps, data, title, tiks_name, ylabel, legend


def main():
    main_labels = ["cross entropy", "focal loss with amp", "focal loss"]
    main_color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    background_color = ['lightsteelblue', 'peachpuff', 'palegreen', 'tab:red', 'tab:purple']

    steps, data, title, tiks_name, ylabel, legend = val_loss()

    plot(steps, data, main_color, background_color, title, main_labels, tiks_name, ylabel, legend)

    steps, data, title, tiks_name, ylabel, legend = val_acc()

    plot(steps, data, main_color, background_color, title, main_labels, tiks_name, ylabel, legend)

    steps, data, title, tiks_name, ylabel, legend = val_jac()

    plot(steps, data, main_color, background_color, title, main_labels, tiks_name, ylabel, legend)


if __name__ == "__main__":
    main()
