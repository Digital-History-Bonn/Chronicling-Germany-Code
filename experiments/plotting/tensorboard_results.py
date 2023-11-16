from csv import reader
from typing import Any, Tuple, List
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

import requests

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

RUNS = [
    ['cross_entropy_amp_A', 'cross_entropy_amp_B', 'cross_entropy_amp_C'],
    ['cross_entropy__no_amp_A', 'cross_entropy__no_amp_B', 'cross_entropy__no_amp_C']
]

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
    data = np.array(list(data_csv)[1:])
    return data[:, 1], data[:, 2]


def get_timeseries(tag: str, runs: List[List[str]]) -> Any:
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
        data.append[]
        for version in run:
            steps, values = get_data(tag, version)
            data[-1].append(values)
        step_lists.append(steps)
    return steps, data


def average(data):
    avg_data = []
    for i in range(0, len(data), 3):
        avg_data.append(np.mean(data[i:i + 3], axis=0))
    return avg_data


def smoothing(data, alpha=.99):
    smoothed_data = []
    for item in data:
        smoothed = []
        curr = item[0]
        for t in item:
            curr = alpha * curr + (1 - alpha) * t
            smoothed.append(curr)

        smoothed_data.append(smoothed)
    return smoothed_data


def smoothing2(data, size=100):
    smoothed_data = []
    for item in data:
        smoothed = []
        for i in range(len(item)):
            smoothed.append(np.mean(item[max(0, i - 50): i]))

        smoothed_data.append(smoothed)
    return smoothed_data


STEPS, EPOCHS = get_timeseries('epoch')


def plot_3d(steps, front, back, title='PLOT', labels=RUNS):
    front_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    back_colors = ['lightsteelblue', 'peachpuff', 'palegreen', 'tab:red', 'tab:purple']

    for i, item in enumerate(back):
        plt.plot_3d(steps, item, color=back_colors[int(i / 3)])

    for i, item in enumerate(front):
        plt.plot_3d(steps, item, label=labels[i], color=front_colors[i])

    plt.title(title)
    plt.xticks(range(0, int(steps[-1]) + 1, int(steps[-1] / 20)), range(0, 101, 5))

    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(f"{title.replace(' ', '')}.pdf", bbox_inches='tight')


steps, data = get_timeseries('train+loss', runs=RUNS[3:])
print(len(steps))
print(len(data[0]))

data_mean = np.mean(data)
data_std = np.std(data)
print(len(data_mean), data_mean[0].shape)
smoothed = smoothing2(data_mean)
print(smoothed[0][:10])
print(smoothed[1][:10])
# plot_3d(steps, smoothed, data, title='Training Loss', labels=['Training mit Pretraining', 'Training ohne Pretraining'])
