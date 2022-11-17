import matplotlib.pyplot as plt
import torch
from PIL import Image

from model import DhSegment
from preprocessing import Preprocessing
from Postprocessing import Postprocess

MODEL = 'models/model.pt'
IN_CHANNELS = 1
OUT_CHANNELS = 2


def plot(image, anno):
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


class Annotator:
    def __init__(self, model_path, in_channels, out_channels):
        """
        Annotates images
        :param model_path: path to trained model
        :param in_channels: channel of the input image
        :param out_channels: number of predicted classes
        """
        self.model = DhSegment([3, 4, 6, 4], in_channels=in_channels, out_channel=out_channels,
                               load_resnet_weights=True)
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

    def annotate(self, pil_image, show=False):
        """
        annotate the given image
        :param pil_image: image HWC
        :param show: bool if image and annotations should be plotted
        :return: json with annotations
        """
        # TODO: add mask functionality
        image, mask = self.pipeline(pil_image)
        image = torch.Tensor([image])
        pred = self.model.forward(image)
        anno = self.postprocess(pred)

        plt.imshow(torch.argmax(pred, dim=1).permute((1, 2, 0)))
        plt.show()

        if show:
            plot(pil_image, anno)

        return anno


if __name__ == '__main__':
    annotator = Annotator(MODEL, IN_CHANNELS, OUT_CHANNELS)
    annotator.annotate(Image.open('data/Validation/Seite0360.JPG'), True)
