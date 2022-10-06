from torch.nn.functional import conv2d
import torch
import numpy as np

# TODO: make output of Postproces json-format


class Postprocess:
    def __init__(self):
        """
        Postprocess of the model output
        """
        self.weights = torch.tensor(np.array([0, 1, -1])[None, None, :, None])

    def __call__(self, pred):
        """
        postprocess prediction
        :param pred: output of the model
        :return: json of annotations
        """
        return self._filter(pred)

    def _filter(self, pred):
        """
        filter lines from Prediction
        :param pred: output of model
        :return: list of all lines
        """
        # TODO: make this also work for multiclass prediction
        pred = torch.argmax(pred, dim=1)
        pred = conv2d(pred, self.weights, bias=None, stride=1, padding='same')
        pred = pred.apply_(lambda x: 1 if x >=1 else 0)
        lines = self._extract_lines(pred)
        return lines

    def _extract_lines(self, pred):
        """
        extract lines from prediction
        :param pred: output of model after filter function
        :return: list of all lines
        """
        lines = []
        height, width = pred.shape[1:]
        done = np.zeros((height, width))

        for x in range(width):
            for y in range(height):
                if done[y, x] != 1:
                    if pred[0, y, x] == 1:
                        lines.append(self._extract_line(x, y, pred, done))
                    done[y, x] = 1

        return lines

    def _extract_line(self, x, y, pred, done):
        """

        :param x: current width position
        :param y: current height positon
        :param pred: output of model after _filter function
        :param done: array of processed pixels
        :return: extracted line
        """
        line = [(y, x)]
        while True:
            next = self._find_next(x, y, pred)
            if next is None:
                line.append((y, x))
                return line

            elif next[0] != y:
                line.append(next)

            done[next] = 1
            y, x = next

    def _find_next(self, x, y, pred):
        """
        if point right from x,y is in lineclass it will be returned else None
        :param x: current width position
        :param y: current height position
        :param pred: output of model after _filter
        :return:
        """
        if x+1 >= pred.shape[2]:
            return None
        if pred[0, y, x+1] == 1:
            return y, x+1

        if pred[0, y+1, x+1] == 1:
            return y+1, x+1

        if pred[0, y-1, x+1] == 1:
            return y-1, x+1

        return None


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from Dataloader import Dataloader
    dataloader = Dataloader('data/Training/', limit=2)
    postprocess = Postprocess()

    X, Y, m = dataloader[0]

    plt.imshow(Y.permute(1, 2, 0), cmap=None, vmin=0, vmax=1, interpolation='none')
    plt.axis('off')
    plt.show()

    lines = postprocess(Y)
    for line in lines:
        xes = [l[1] for l in line]
        yelons = [l[0] for l in line]
        plt.plot(xes, yelons, color='r')

    plt.imshow(X.permute(1, 2, 0))
    plt.axis('off')
    plt.show()
