"""Utility Module"""
import numpy as np
import matplotlib.pyplot as plt  # type: ignore
import torch.nn.functional as f
import torch
import PIL.Image as Image  # type: ignore
import sklearn  # type: ignore
from numpy import ndarray


def multi_class_csi(pred: torch.Tensor, true: torch.Tensor, classes: int = 10) -> ndarray:
    """Calculate csi score using true positives, true negatives and false negatives from confusion matrix.
     Csi score is used as substitute for accuracy, calculated separately for each class.
     Returns numpy array with an entry for every class. If every prediction is a true negative,
     the score cant be calculated and the array will contain nan. These cases should be completely ignored.
     :param pred: prediction tensor
     :param true: target tensor
     :param classes: number of possible classes
     :return:
     """
    pred = pred.flatten()
    true = true.flatten()
    matrix = sklearn.metrics.confusion_matrix(true, pred, labels=range(0, classes))
    true_positive = np.diagonal(matrix)
    false_positive = np.sum(matrix, axis=1) - true_positive
    false_negative = np.sum(matrix, axis=0) - true_positive
    return np.array(true_positive / (true_positive + false_negative + false_positive))


def step(x):
    return 1 if x > 0 else 0


step = np.vectorize(step)


def plot_sampels(dataloader, n, model=None, title='Samples'):
    images, target, preds = [], [], []
    for i in np.random.randint(0, len(dataloader), n):
        x, y, _ = dataloader[i]
        images.append(x)
        if dataloader.channel > 1:
            y = f.one_hot(y, num_classes=dataloader.channel)
        target.append(y)
        if model is not None:
            pred = model(x[None, :]).detach()[0]
            pred = np.argmax(pred, axis=0)[None, :]
            preds.append(pred)
        else:
            preds = None

    plot(images, target, preds, title=title)


def plot(images, target, preds=None, title='Samples'):
    assert len(images) == len(
        target), f"list of images not the same lenght as targets got {len(images)} and {len(target)}"
    n = len(images)

    columns = [images, target]
    if preds is not None:
        columns.append(preds)

    img_title = ['image', 'target', 'prediction']
    cmap = {0: 'gray', 1: None, 2: None}

    fig = plt.figure(figsize=(15, 15))
    fig.suptitle(title, fontsize=26)
    for s in range(n):
        for c, source in enumerate(columns):
            i = s * len(columns) + c
            fig.add_subplot(n, len(columns), i + 1)
            plt.title(img_title[c], fontsize=18)
            plt.imshow(source[s].float().permute(1, 2, 0), cmap=cmap[c], vmin=0, vmax=1, interpolation='none')
            plt.axis('off')
    plt.show()


def plot_process(images, targets, preds, results, title='process'):
    n = len(images)
    processes = [(im, tar, pred, res) for im, tar, pred, res in zip(images, targets, preds, results)]
    img_title = ['input', 'target', 'prediction', 'result']

    fig = plt.figure(figsize=(15, 15))
    fig.suptitle(title, fontsize=26)
    for s in range(n):
        proc = processes[0]
        for c in range(4):
            i = s * 4 + c
            fig.add_subplot(n, 4, i + 1)
            plt.title(img_title[c], fontsize=18)
            plt.imshow(proc[c])
            plt.axis('off')
    plt.show()


def replace_substrings(string, replacements):
    for i, j in replacements.items():
        string = string.replace(i, j)
    return string


def get_file(file: str, scale=0.25) -> torch.Tensor:
    """
    loads a image as tensor
    :param file: path to file
    :param scale: scale
    :return: image as torch.Tensor
    """
    img = Image.open(file).convert('RGB')
    shape = int(img.size[0] * scale), int(img.size[1] * scale)
    img = img.resize(shape, resample=Image.BICUBIC)

    w_pad, h_pad = (32 - (shape[0] % 32)), (32 - (shape[1] % 32))
    img_np = np.pad(np.asarray(img), ((0, h_pad), (0, w_pad), (0, 0)), 'constant', constant_values=0)
    img_t = np.transpose(torch.tensor(img_np), (2, 0, 1))
    return torch.unsqueeze(torch.tensor(img_t / 255, dtype=torch.float), dim=0)


class RollingAverage:
    def __init__(self, length, round=4):
        self.length = length
        self.round = round
        self.values = np.zeros(length)
        self.t = -1

    def __call__(self, new_value):
        self.t += 1
        self.values[self.t % self.length] = new_value
        return np.round(np.average(self.values[:min(self.t + 1, self.length)]), self.round)
