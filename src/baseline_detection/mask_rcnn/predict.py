from pathlib import Path
from typing import Dict, Union, List

import matplotlib.pyplot as plt
import torch
from torchvision.transforms import GaussianBlur
from skimage import io
from skimage.draw import line as sk_line
from torchvision.models.detection import MaskRCNN

from src.baseline_detection.mask_rcnn.dataset_baseline import CustomDataset
from src.baseline_detection.mask_rcnn.postprocessing import postprocess
from src.baseline_detection.mask_rcnn.preprocess import extract
from src.baseline_detection.mask_rcnn.trainer_textline import get_model
from monai.networks.nets import BasicUNet



def prior(size: int):
    p1 = int(size * 0.3)
    p2 = int(size * 0.6)
    p3 = int(size * 0.80)
    return torch.hstack([torch.zeros(p1),
                         torch.linspace(0.0, 1, p2 - p1),
                         torch.ones(p3 - p2),
                         torch.linspace(1, 0.0, size - p3)])


def draw_lines_on_image(image_tensor, lines):
    # Convert torch tensor to numpy array
    image_np = image_tensor.permute(1, 2, 0).numpy()

    # Draw lines on the image
    for line in lines:
        # Iterate over consecutive pairs of points in the line
        for i in range(len(line) - 1):
            # Extract coordinates
            y1, x1 = line[i]
            y2, x2 = line[i + 1]

            # Generate coordinates for the line using skimage.draw.line
            rr, cc = sk_line(y1, x1, y2, x2)

            # Draw the line
            image_np[rr, cc] = [0, 0, 255]  # Color of the line (here, blue)

    return image_np.transpose((2, 0, 1))


def predict_baseline(box: torch.Tensor, mask: torch.Tensor, map: torch.Tensor):
    line_region = map * mask[0]
    line_region = line_region[box[1]:box[3], box[0]:box[2]] * prior(box[3] - box[1])[:, None]

    y_pos = torch.argmax(line_region, dim=0)
    y_values = torch.amax(line_region, dim=0)

    line = torch.vstack([box[1] + y_pos, torch.arange(box[0], box[0] + len(y_pos))]).T[y_values > 0.5]

    return line.int()


def predict_image(textline_model: MaskRCNN,
                  baseline_model: BasicUNet,
                  image: torch.Tensor,
                  device: torch.device):

    gauss_filter = GaussianBlur(kernel_size=5, sigma=2.0)
    image = image.to(device)

    # predict example form training set
    print(f"{image.shape=}")
    pred: Dict[str, Union[torch.Tensor, List[torch.Tensor]]] = textline_model([image])[0]

    # move predictions to cpu
    pred["boxes"] = pred["boxes"].detach().cpu()
    pred["labels"] = pred["labels"].detach().cpu()
    pred["scores"] = pred["scores"].detach().cpu()
    pred["masks"] = pred["masks"].detach().cpu()

    # postprecess image (non maxima supression)
    pred = postprocess(pred, method='iom', threshold=.6)

    pred["boxes"] = pred["boxes"].int()
    pred["masks"] = pred["masks"] > 0.5

    baseline_probability_map = baseline_model(image[None])[0, 1]

    baseline_probability_map = baseline_probability_map.detach().cpu()
    baseline_probability_map = gauss_filter(baseline_probability_map[None])[0]

    pred['lines'] = []
    for box, mask in zip(pred["boxes"], pred["masks"]):
        line = predict_baseline(box, mask, baseline_probability_map)
        pred['lines'].append(line)

    return pred


def predict_page(image: torch.Tensor, annotation: List[Dict[str, List[torch.Tensor]]]):

    device = torch.device('cuda:0')

    textline_model = get_model(objective='textlines',
                               load_weights='MaskRCCNLineDetection2_Newspaper_textlines_e25_es')
    textline_model.to(device)
    textline_model.eval()

    baseline_model = BasicUNet(spatial_dims=2, in_channels=3, out_channels=2)
    baseline_model.load_state_dict(torch.load(f'{Path(__file__).parent.absolute()}/../../models/test3_baseline_e100_es.pt'))
    baseline_model.to(device)
    baseline_model.eval()

    prediction = {"boxes": torch.zeros((0, 4)),
                  "scores": torch.zeros((0,)),
                  "masks": torch.zeros((0, *image.shape)),
                  "lines": []}

    for region in annotation:
        subimage = image[:, region['part'][0]: region['part'][2],
                   region['part'][1]: region['part'][3]]

        pred = predict_image(textline_model, baseline_model, subimage, device)

        shift = torch.tensor([region['part'][1], region['part'][0], region['part'][1], region['part'][0]])
        prediction["boxes"] = torch.vstack([prediction["boxes"], pred["boxes"] + shift])
        prediction["scores"] = torch.hstack([prediction["scores"], pred["scores"]])
        # prediction["masks"] = torch.stack([prediction["masks"], pred["masks"] + shift])
        shift = torch.tensor([region['part'][0], region['part'][1]])
        prediction["lines"].extend([l + shift for l in pred["lines"]])

    return prediction


def main():
    device = torch.device('cuda:0')

    textline_model = get_model(objective='textlines',
                               load_weights='MaskRCCNLineDetection2_Newspaper_textlines_e25_es')
    textline_model.to(device)
    textline_model.eval()

    baseline_model = BasicUNet(spatial_dims=2, in_channels=3, out_channels=2)
    baseline_model.load_state_dict(torch.load(f'{Path(__file__).parent.absolute()}/../../models/test3_baseline_e100_es.pt'))
    baseline_model.to(device)
    baseline_model.eval()

    dataset = CustomDataset(f'{Path(__file__).parent.absolute()}/../../data/Newspaper/valid')

    image, _ = dataset[6]

    pred = predict_image(textline_model, baseline_model, image, device)

    result = draw_prediction(image, pred)

    result = draw_lines_on_image(result, pred['lines'])

    print(f"{result.shape=}, {result.max()}, {result.min()}, {result.dtype=}")

    io.imsave(f'{Path(__file__).parent.absolute()}/../../data/Examples/baselineDetectionExample6.png', result.transpose((1, 2, 0)))


if __name__ == '__main__':
    image = torch.tensor(io.imread(f"{Path(__file__).parent.absolute()}/../../data/newspaper-dataset-main-images/images/Koelnische_Zeitung_1924 - 0085.jpg")).permute(2, 0, 1)
    image = image.to(torch.device('cuda:0'))
    image = image.float()

    anno = extract(f"{Path(__file__).parent.absolute()}/../../data/newspaper-dataset-main-annotations/annotations/Koelnische_Zeitung_1924 - 0085.xml")

    pred = predict_page(image, anno)

    del pred['masks']

    result = draw_prediction(image.cpu(), pred)

    result = draw_lines_on_image(result, pred['lines'])

    io.imsave(
        f'{Path(__file__).parent.absolute()}/../../data/Examples/baselineDetectionPageExample6.png',
        result.transpose((1, 2, 0)))