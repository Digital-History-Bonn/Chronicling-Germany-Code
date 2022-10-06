import matplotlib.pyplot as plt
import torch
from PIL import Image

from model import dhSegment
from Preprocessing import Preprocessing
from Postprocessing import Postprocess

MODEL = 'models/model.pt'
IN_CHANNELS = 1
OUT_CHANNELS = 2


class Annotator:
    def __init__(self, model_path, in_channels, out_channels):
        """
        Annotates images
        :param model_path: path to trained model
        :param in_channels: channel of the input image
        :param out_channels: number of predicted classes
        """
        self.model = dhSegment([3, 4, 6, 4], in_channels=in_channels, out_channel=out_channels, load_resnet_weights=True)
        self.model = self.model.float()
        self.model.load(model_path)

        self.pipeline = Preprocessing()
        self.postprocess = Postprocess()

    def __getitem__(self, image):
        """
        get annotation for given image
        :param image: image HWC
        :return: json with annotations
        """
        return self.annotate(image)

    def annotate(self, image, show=False):
        """
        annotate the given image
        :param image: image HWC
        :param show: bool if image and annotations should be plotted
        :return: json with annotations
        """
        # TODO: add mask functionality
        input, mask = self.pipeline(image)
        input = torch.Tensor([input])
        pred = self.model.forward(input)
        anno = self.postprocess(pred)

        plt.imshow(torch.argmax(pred, dim=1).permute((1, 2, 0)))
        plt.show()

        if show:
            self.plot(image, anno)

        return anno

    def plot(self, image, anno):
        """
        plot function for image and annotations
        :param image: Image
        :param anno: annotations of that image
        :return: None
        """
        for line in anno:
            x = [l[1] for l in line]
            y = [l[0] for l in line]
            plt.plot(x, y, color='r', linewidth=1)

        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    annotator = Annotator(MODEL, IN_CHANNELS, OUT_CHANNELS)
    image = Image.open('data/Validation/Seite0360.JPG')
    annotator.annotate(image, True)

