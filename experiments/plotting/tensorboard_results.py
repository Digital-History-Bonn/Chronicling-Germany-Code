import json
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
        'multi-acc-Test/class+0',
        'multi-acc-Test/class+1',
        'multi-acc-Test/class+2',
        'multi-acc-Test/class+3',
        'multi-acc-Test/class+4',
        'multi-acc-Test/class+5',
        'multi-acc-Test/class+6',
        'multi-acc-Test/class+7',
        'multi-acc-Test/class+8',
        'multi-acc-Test/class+9',
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
        'Test/accuracy',
        'Test/jaccard+score',
        'Test/loss',
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
#     ['dh_segment_euro_B', 'dh_segment_euro_C', 'dh_segment_euro_D'],
#     ['cbam_euro_no_skip_A', 'cbam_euro_no_skip_B', 'cbam_euro_no_skip_C'],
#     ['trans_unet_fast_euro_A', 'trans_unet_fast_euro_B', 'trans_unet_fast_euro_C']
# ]

# RUNS = [
#     ['lerning_rate_Test_4_6_A', 'lerning_rate_4_6_B', 'lerning_rate_4_6_C'],
#     ['scaling_64_512_1.0', 'scaling_64_512_2.0', 'scaling_64_512_3.0'],
#     ['reduce_1.0', 'reduce_2.0', 'reduce_3.0'],
# ]

# RUNS = [
#     ['dh_segment_newspaper_H'],
#     ['cbam_newspaper_A'],
#     ['trans_unet_newspaper_A']
# ]
# RUNS = [
#     ['dh_segment_newspaper_H'],
# ]
# RUNS = [
#     ['cbam_newspaper_A'],
# ]
# RUNS = [
#     ['trans_unet_newspaper_A']
# ]

RUNS = [
    ['neurips-09-2024_scratch_1.0_eval', 'neurips-09-2024_scratch_2.0_eval', 'neurips-09-2024_scratch_3.0_eval',
     'neurips-09-2024_scratch_4.0_eval'],
    ['neurips-09-2024_1.0_eval', 'neurips-09-2024_2.0_eval', 'neurips-09-2024_3.0_eval', 'neurips-09-2024_4.0_eval']
]

# RUNS = [['lerning_rate_Test_4_6_A', 'lerning_rate_4_6_B', 'lerning_rate_4_6_C']]


# RUNS = [['final_dh_1.0', 'final_dh_2.0', 'final_dh_3.0']]
# RUNS = [['final_dh_segment_1.0', 'final_dh_segment_2.0', 'final_dh_segment_3.0'],
#         ['final_cbam_1.0', 'final_cbam_2.0', 'final_cbam_3.0']]

XTICKS = [np.arange(1, 11) + 0.375, ["Background",
                                     "Caption",
                                     "Table",
                                     "Paragraph",
                                     "Heading",
                                     "Header",
                                     "Separator Vertical",
                                     "Separator Horizontal",
                                     "Image",
                                     "Inverted Text"]]

# XTICKS = [np.arange(1, 9) + 0.375, ["Background",
#                                      "Caption",
#                                      "Table",
#                                      "Article",
#                                      "Heading",
#                                      "Header",
#                                      "Separator",
#                                      "Separator Horizontal"]]

# XTICKS = [np.arange(1, 5) + 0.375, ["Background",
#                                      "Article",
#                                      "Heading",
#                                      "Separator"]]

# XTICKS = [np.arange(1, 5) + 0.375, ["Background",
#                                      "Text",
#                                      "Separator",
#                                      "Separator Big"]]

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
    data = []
    step_lists = []
    for run in runs:
        value_list = []
        for version in run:
            steps, values = get_data(tag, version)
            value_list.append(values)
        data.append(np.array(value_list))
        step_lists.append(np.array(steps, dtype=int))

    # custom step correction
    # step_lists[-1] = step_lists[-1]//2
    # step_lists[-2] = step_lists[-2]//2

    return step_lists, data


def average(data):
    avg_data = []
    for i in range(0, len(data), 3):
        avg_data.append(np.mean(data[i:i + 3], axis=0))
    return avg_data


STEPS, EPOCHS = get_timeseries('epoch')


def plot_bar(data: ndarray, stds: ndarray, name: str, ticks: Any, labels: List[str], title: str, ylabel: str) -> None:
    """
    Plot 2d data, which has been summarized from 3d Data along one axis.
    :param data: ndarray 2d data
    :param stds: standart deviation ndarray fro each data point
    :param name: name for saved file
    :param xlabel: name for x-axis label
    """
    xdata = np.arange(data.shape[0]) + 1
    fig, axplt = plt.subplots()
    axplt.set_ylabel(ylabel)

    for i in range(2):
        axplt.bar(xdata + i / 4, data[:, i], 0.25, align="center", yerr=stds[:, i], label=labels[i])
        # axplt.bar(xdata + i / 4, data[:, i], 0.25, align="center", label=labels[i])
    # axplt.bar(xdata, data, align="center", yerr=stds)
    # axplt.bar(xdata, data, align="center")

    axplt.set_xticks(ticks[0], ticks[1], rotation=45, ha='right')
    axplt.legend(loc="upper right")
    fig.subplots_adjust(bottom=0.2)
    plt.title(title)
    plt.grid()

    plt.savefig(f"{name}.pdf")

    # pylint: disable=assignment-from-no-return
    fig = plt.gcf()
    fig = tikzplotlib_fix_ncols(fig)
    tikzplotlib.save(f"{name}.tex")
    plt.show()


def plot(steps, data, main_color, background_color, title, labels, tiks_name, ylabel, legend):
    """Plots timeseries with error bands"""

    # custom step correction
    epochs = EPOCHS[1][0].astype(int)

    steps[1] = correct_from(epochs, steps)

    data[1] = np.stack([data[1][0][:911], np.append(data[1][1], [data[1][1][-1], data[1][1][-1]]), data[1][2][:911]])
    data[0] = data[0][:, : 911]

    for index, timeseries in enumerate(data):
        mean = np.mean(timeseries, axis=0)
        error = np.std(timeseries, axis=0)
        # mean = timeseries[0]

        plt.plot(steps[index], mean, color=main_color[index], label=labels[index])
        # plt.plot(steps[index], mean, color=main_color[index])
        plt.fill_between(steps[index], mean - error, mean + error, color=background_color[index])
    plt.title(title)

    # set_xticks(steps, epochs)
    set_xticks_per_version(0, steps, epochs)

    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.grid()
    plt.legend(loc=legend)

    plt.savefig(f"{tiks_name}.pdf")
    fig = plt.gcf()
    fig = tikzplotlib_fix_ncols(fig)
    tikzplotlib.clean_figure()
    tikzplotlib.save(f"{tiks_name}.tex")
    plt.clf()
    # plt.show()


def correct_from(epochs, steps):
    """Corrects steps number from epoch_number onwards. In this case the steps in the second run from epoch 200 forward
    rise twice as fast as in the first one. Therefore from epoch 200 they are corrected to match the other run."""
    epoch_number = 200
    index = np.where(epochs == epoch_number)[0][0]
    return np.concatenate([steps[1][: index], steps[1][index] + (steps[1][index:] - steps[1][index]) // 2])[:-1]


def set_xticks(steps, epochs=EPOCHS[0][0].astype(int)):
    """Arange x ticks so that the units is epochs and not steps. Calculates step per value based on last epoch and
    last step. This only works if this does not change throughout training and versions."""
    number_epochs = epochs[-1]
    number_steps = steps[1][-1]
    step_per_epoch = number_steps // number_epochs
    epoch_tiks = 50
    plt.xticks(np.append(np.arange(0, number_steps, step_per_epoch * epoch_tiks)[:-1], steps[1][-1] + 1),
               np.arange(0, number_epochs + 1, epoch_tiks))


def set_xticks_per_version(index, steps, epochs):
    """Arange x ticks so that the units is epochs and not steps. This version do"""
    # epochs = EPOCHS[index][0].astype(int)
    # steps = STEPS[index]
    number_epochs = epochs[-1]
    steps = correct_from(epochs, np.array(STEPS))
    # insert missing epoch values
    # missing = []
    # for index, _ in enumerate(epochs):
    #     if index + 1 < len(epochs) and epochs[index] != epochs[index + 1] and epochs[index] + 1 != epochs[index + 1]:
    #         missing.append(index)
    #
    # missing = np.array(missing)
    # epochs = np.array(epochs).astype(int)
    # steps = np.array(steps).astype(int)
    # epochs = np.insert(epochs, missing + 1, epochs[missing] + 1)
    # steps = np.insert(steps, missing + 1, steps[missing])

    change = np.insert(epochs[1:] != epochs[:-1], 0, True)[:-2]
    steps_epoch = np.insert(steps[change], 0, 0)
    epoch_tiks = 50
    plt.xticks(np.append(steps_epoch[1::epoch_tiks], steps[-1] + 1), np.arange(0, number_epochs + 1, epoch_tiks))


def val_loss():
    steps, data = get_timeseries('val/loss')
    title = "Validation Loss"
    tiks_name = "final_val_loss"
    ylabel = "Loss"
    legend = "upper right"

    return steps, data, title, tiks_name, ylabel, legend


def val_acc():
    steps, data = get_timeseries('val/accuracy')
    title = "Validation accuracy"
    tiks_name = "final_val_acc"
    ylabel = "Accuracy"
    legend = "lower right"

    return steps, data, title, tiks_name, ylabel, legend


def val_jac():
    steps, data = get_timeseries('val/jaccard score')
    title = "Validation Jaccard Score"
    tiks_name = "final_val_jac"
    ylabel = "Jaccard Score"
    legend = "lower right"

    return steps, data, title, tiks_name, ylabel, legend


def graph():
    main_labels = ['DhSegment', 'DhSegment CBAM']
    main_color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    background_color = ['lightsteelblue', 'peachpuff', 'palegreen', 'tab:red', 'tab:purple']

    steps, data, title, tiks_name, ylabel, legend = val_loss()

    plot(steps, data, main_color, background_color, title, main_labels, tiks_name, ylabel, legend)

    steps, data, title, tiks_name, ylabel, legend = val_acc()

    plot(steps, data, main_color, background_color, title, main_labels, tiks_name, ylabel, legend)

    steps, data, title, tiks_name, ylabel, legend = val_jac()

    plot(steps, data, main_color, background_color, title, main_labels, tiks_name, ylabel, legend)


def class_sci():
    tags = ['multi-acc-Test/class 0',
            'multi-acc-Test/class 1',
            'multi-acc-Test/class 2',
            'multi-acc-Test/class 3',
            'multi-acc-Test/class 4',
            'multi-acc-Test/class 5',
            'multi-acc-Test/class 6',
            'multi-acc-Test/class 7',
            'multi-acc-Test/class 8',
            'multi-acc-Test/class 9']

    # tags = ['multi-acc-Test/class 0',
    #         'multi-acc-Test/class 4',
    #         'multi-acc-Test/class 5',
    #         'multi-acc-Test/class 7',]

    # tags = ['multi-acc-Test/class 0',
    #         'multi-acc-Test/class 4',
    #         'multi-acc-Test/class 7',
    #         'multi-acc-Test/class 9',]

    # tags = ['multi-acc-Test/class 0',
    #         'multi-acc-Test/class 2',
    #         'multi-acc-Test/class 3',
    #         'multi-acc-Test/class 4',
    #         'multi-acc-Test/class 5',
    #         'multi-acc-Test/class 6',
    #         'multi-acc-Test/class 7',
    #         'multi-acc-Test/class 9']

    mean, std = get_data_bar(tags)

    # labels = ["DhSegment", "CBAM", "Trans Unet"]
    # labels = ["DhSegment", "CBAM skip", "CBAM no Skip"]
    # labels = ["so scaling", "scaling", "reduce", "reduce_focal"]
    labels = ["scratch", "pretrained"]

    name = "final-class-csi"
    title = "Multi Class IoU"
    ylabel = 'IoU'

    # plot_bar(data, np.zeros(1), name, XTICKS, labels, title)
    # plot_bar(mean, std, name, XTICKS, labels, title, ylabel)
    return mean.tolist(), std.tolist()


def class_precision():
    tags = ['multi-precision-Test/class 0',
            'multi-precision-Test/class 1',
            'multi-precision-Test/class 2',
            'multi-precision-Test/class 3',
            'multi-precision-Test/class 4',
            'multi-precision-Test/class 5',
            'multi-precision-Test/class 6',
            'multi-precision-Test/class 7',
            'multi-precision-Test/class 8',
            'multi-precision-Test/class 9']

    # tags = ['multi-precision-Test/class 0',
    #         'multi-precision-Test/class 4',
    #         'multi-precision-Test/class 5',
    #         'multi-precision-Test/class 7']
    #
    # tags = ['multi-precision-Test/class 0',
    #         'multi-precision-Test/class 4',
    #         'multi-precision-Test/class 7',
    #         'multi-precision-Test/class 9']

    # tags = ['multi-precision-Test/class 0',
    #         'multi-precision-Test/class 2',
    #         'multi-precision-Test/class 3',
    #         'multi-precision-Test/class 4',
    #         'multi-precision-Test/class 5',
    #         'multi-precision-Test/class 6',
    #         'multi-precision-Test/class 7',
    #         'multi-precision-Test/class 9']

    mean, std = get_data_bar(tags)

    labels = ["scratch", "pretrained"]
    name = "final-class-precision"
    title = "Multi Class Precision"
    ylabel = 'Precision'

    # plot_bar(data, np.zeros(1), name, XTICKS, labels, title)
    # plot_bar(mean, std, name, XTICKS, labels, title, ylabel)
    return mean.tolist(), std.tolist()


def get_data_bar(tags):
    data = []
    for tag in tags:
        _, values = get_timeseries(tag)
        tag_data = []

        for i in range(len(values)):
            tag_data.append([values[i][j][-1] for j in range(len(values[i]))])

        # data.append([values[i][0][-1] for i in range(3)])
        # data.append(values[0][0][-1])
        data.append(tag_data)
        mean = []
        std = []
    for version in data:
        mean.append([np.mean(version[i]) for i in range(len(version))])
        std.append([np.std(version[i]) for i in range(len(version))])
    return np.array(mean), np.array(std)


def class_recall():
    tags = ['multi-recall-Test/class 0',
            'multi-recall-Test/class 1',
            'multi-recall-Test/class 2',
            'multi-recall-Test/class 3',
            'multi-recall-Test/class 4',
            'multi-recall-Test/class 5',
            'multi-recall-Test/class 6',
            'multi-recall-Test/class 7',
            'multi-recall-Test/class 8',
            'multi-recall-Test/class 9']

    # tags = ['multi-recall-Test/class 0',
    #         'multi-recall-Test/class 4',
    #         'multi-recall-Test/class 5',
    #         'multi-recall-Test/class 7',]

    # tags = ['multi-recall-Test/class 0',
    #         'multi-recall-Test/class 4',
    #         'multi-recall-Test/class 7',
    #         'multi-recall-Test/class 9',]

    # tags = ['multi-recall-Test/class 0',
    #         'multi-recall-Test/class 2',
    #         'multi-recall-Test/class 3',
    #         'multi-recall-Test/class 4',
    #         'multi-recall-Test/class 5',
    #         'multi-recall-Test/class 6',
    #         'multi-recall-Test/class 7',
    #         'multi-recall-Test/class 9']

    mean, std = get_data_bar(tags)

    # labels = ["so scaling", "scaling", "reduce", "reduce_focal"]
    labels = ["scratch", "pretrained"]
    name = "final-class-recall"
    title = "Multi Class Recall"
    ylabel = "Recall"

    return mean.tolist(), std.tolist()
    # plot_bar(data, np.zeros(1), name, XTICKS, labels, title)
    # plot_bar(mean, std, name, XTICKS, labels, title, ylabel)


def class_f1():
    # dataset = "Test"
    dataset = "Out of distribution test"
    tags = [f'multi-f1-{dataset}/class 0',
            f'multi-f1-{dataset}/class 1',
            f'multi-f1-{dataset}/class 2',
            f'multi-f1-{dataset}/class 3',
            f'multi-f1-{dataset}/class 4',
            f'multi-f1-{dataset}/class 5',
            f'multi-f1-{dataset}/class 6',
            f'multi-f1-{dataset}/class 7',
            f'multi-f1-{dataset}/class 8',
            f'multi-f1-{dataset}/class 9']

    mean, std = get_data_bar(tags)

    # labels = ["so scaling", "scaling", "reduce", "reduce_focal"]
    labels = ["scratch", "pretrained"]
    name = "final-class-recall"
    title = "Multi Class Recall"
    ylabel = "Recall"

    return mean.tolist(), std.tolist()
    # plot_bar(data, np.zeros(1), name, XTICKS, labels, title)
    # plot_bar(mean, std, name, XTICKS, labels, title, ylabel)


def results():
    tags = ['Test/loss',
            'Test/accuracy',
            'Test/jaccard score']

    mean, std = get_data_bar(tags)

    xticks = [np.arange(1, 4) + 0.375, ["Loss",
                                        "Accuracy",
                                        "Jaccard Score"]]
    labels = ["scratch", "pretrained"]
    name = "final-results"
    title = "Test Results"
    ylabel = ""

    # plot_bar(data, np.zeros(1), name, xticks, labels, title)
    plot_bar(mean, std, name, xticks, labels, title, ylabel)


def bar():

    # results()
    results = []

    # results.append(class_sci())
    # print("iou")
    # results.append(class_precision())
    # print("precision")
    # results.append(class_recall())
    # print("recall")
    results.append(class_f1())
    print("f1")

    with open('eval_results.json', 'w', encoding='utf8') as json_file:
        json.dump(results, json_file)


def main():
    # graph()
    bar()


if __name__ == "__main__":
    main()
