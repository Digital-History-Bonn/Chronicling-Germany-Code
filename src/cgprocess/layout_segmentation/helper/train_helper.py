"""Module for utility function for training. This includes initialization of the model, dataset and dataloaders, as
well as computing metrics for validation and test."""

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix

from cgprocess.layout_segmentation.class_config import TOLERANCE, PADDING_LABEL
from cgprocess.layout_segmentation.models.dh_segment_wide import DhSegmentWide
from cgprocess.layout_segmentation.datasets.train_dataset import CropDataset
from cgprocess.layout_segmentation.models.dh_segment import DhSegment
from cgprocess.layout_segmentation.models.dh_segment_2 import DhSegment2
from cgprocess.layout_segmentation.models.dh_segment_cbam import DhSegmentCBAM
from cgprocess.layout_segmentation.models.dh_segment_small import DhSegmentSmall
from cgprocess.layout_segmentation.models.trans_unet import VisionTransformer
from cgprocess.layout_segmentation.processing.preprocessing import Preprocessing
from cgprocess.layout_segmentation.train_config import IN_CHANNELS, OUT_CHANNELS
from cgprocess.layout_segmentation.utils import adjust_path
from cgprocess.shared.datasets import PageDataset
from cgprocess.shared.utils import get_file_stem_split


def init_model(
        load: Union[str, None],
        device: str,
        model_str: str,
        freeze: bool = True,
        skip_cbam: bool = False,
) -> Any:
    """
    Initialise model
    :param overwrite_load_channels: overwrites standart output channels, if a loaded model has a different size than
    intended for training or prediction
    :param load: contains path to load the model from. If False, the model will be initialised randomly
    :param model_str: contains model architecture name
    :param freeze: activates encoder freezing
    :param skip_cbam: activates cbam skip connection
    :return: loaded model
    """
    if model_str == "dh_segment":
        # create model
        model: Any = DhSegment(
            [3, 4, 6, 4],
            in_channels=IN_CHANNELS,
            out_channel=OUT_CHANNELS,
            load_resnet_weights=True,
        )
        model = setup_dh_segment(device, load, model, freeze)
    elif model_str == "trans_unet":
        load_backbone = not load
        model = VisionTransformer(
            load_backbone=load_backbone,
            in_channels=IN_CHANNELS,
            out_channel=OUT_CHANNELS,
            load_resnet_weights=True,
        )

        model = model.float()
        if freeze:
            model.encoder.freeze_encoder()
        # load model if argument is None, it does nothing
        model.load(load, device)

        model.encoder.means = torch.tensor((0.485, 0.456, 0.406))
        model.encoder.stds = torch.tensor((0.229, 0.224, 0.225))
    elif model_str == "dh_segment_cbam":
        model = DhSegmentCBAM(
            in_channels=IN_CHANNELS,
            out_channel=OUT_CHANNELS,
            load_resnet_weights=True,
            cbam_skip_connection=skip_cbam,
        )
        model = setup_dh_segment(device, load, model, freeze)
    elif model_str == "dh_segment_small":
        model = DhSegmentSmall(
            in_channels=IN_CHANNELS, out_channel=OUT_CHANNELS, load_resnet_weights=True
        )
        model = setup_dh_segment(device, load, model, freeze)
    elif model_str == "dh_segment_2":
        model = DhSegment2(
            in_channels=IN_CHANNELS, out_channel=OUT_CHANNELS, load_resnet_weights=True
        )
        model = setup_dh_segment(device, load, model, freeze)
    elif model_str == "dh_segment_wide":
        model = DhSegmentWide(
            in_channels=IN_CHANNELS, out_channel=OUT_CHANNELS, load_resnet_weights=True
        )
        model = setup_dh_segment(device, load, model, freeze)
    assert model, "No valid model string supplied in model parameter"
    model.to(device)
    return model


def setup_dh_segment(
        device: str, load: Union[str, None], model: Any, freeze: bool
) -> Any:
    """
    Setup function for dh_segment and dh_segment_cbam
    :param override_load_channels: overrides the out channel number with that a model will be loaded.
    :param freeze: freezes the encoder
    :param device:
    :param load: contains path to load the model from. If False, the model will be initialised randomly
    :param model:
    :return:
    """

    assert model.out_channel == OUT_CHANNELS, \
        "Config Error. Output Classes of the model have to match OUT_CHANNELS constant."

    assert len(
        TOLERANCE) == OUT_CHANNELS-1, \
        ("Number of tolerance values for classes need to have exactly one value less than the OUT_CHANNELS "
         "constant, ignoring the background class.")

    model = model.float()
    if freeze:
        model.freeze_encoder()
    # load model if argument is None, it does nothing
    model.load(load, device)
    # if override_load_channels != OUT_CHANNELS:
    #     model.conv2 = conv1x1(32, 12)
    #     model.out_channel = 12
    #     print(
    #         f"overriding model loading out channels. {override_load_channels} channels are "
    #         f"loaded and overwritten with {OUT_CHANNELS} channels")
    # set mean and std in a model for normalization
    model.means = torch.tensor((0.485, 0.456, 0.406))
    model.stds = torch.tensor((0.229, 0.224, 0.225))
    return model


def load_score(
        load: Union[str, None], args: argparse.Namespace
) -> Tuple[float, int, int]:
    """
    Load the score corresponding to the loaded model if requestet, as well as the step value to continue logging.
    """
    best_score = 1000
    step = 0
    epoch = 1
    if args.load_score and load:
        with open(f"scores/model_{args.load}.json", "r", encoding="utf-8") as file:
            best_score, step, epoch = json.load(file)
    return best_score, step, epoch


def focal_loss(
        prediction: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor,
        gamma: float = 2.0,
) -> torch.Tensor:
    """
    Calculate softmax focal loss. Migrated from https://github.com/google-deepmind/optax/pull/573/files
    :param weights: tensor of size num_classes with weights between 0 and 1
    :param prediction: Unnormalized log probabilities, with shape `[batch, num_classes, ...]`.
    :param target: Valid probability distributions (non-negative, sum to 1), e.g a
      one hot encoding specifying the correct class for each input
    :param gamma: Focusing parameter `>=0`. It controls the contribution of higher confidence predictions.
      Defaults to 2.
    :return: loss tensor [batches, ...]"""
    probalbilities = torch.nn.functional.softmax(prediction, dim=1)
    probalbilities = target * probalbilities + (1 - target) * (1 - probalbilities)
    focus = torch.pow(1 - probalbilities, gamma)
    loss = - weights[None, :, None, None] * focus * torch.log(probalbilities + 1e-15)
    return torch.sum(loss, dim=1) / OUT_CHANNELS  # type: ignore


def calculate_scores(data: torch.Tensor) -> Tuple[float, float, Tensor]:
    """
    Applies argmax on the channel dimension of the prediction to obtain int class values.
    Then computes jaccard, accuracy and multi class csi score (critical sucess index)
    :param data: Combined and flattened prediction and Target with shape [P, C] P being number of pixels.
    The last Channel contains target data.
    """
    pred = data[:, :-1]
    targets = torch.squeeze(data[:, -1].to(torch.uint8))

    jaccard_fun = JaccardIndex(
        task="multiclass", num_classes=OUT_CHANNELS, average="weighted", ignore_index=PADDING_LABEL
    ).to(
        pred.device
    )  # type: ignore
    accuracy_fun = MulticlassAccuracy(num_classes=OUT_CHANNELS, average="weighted", ignore_index=PADDING_LABEL).to(
        pred.device
    )
    confusion_metric = MulticlassConfusionMatrix(num_classes=OUT_CHANNELS, ignore_index=PADDING_LABEL).to(
        pred.device
    )

    pred = torch.argmax(pred, dim=1).type(torch.uint8)

    # pylint: disable=not-callable
    jaccard = jaccard_fun(pred.flatten(), targets.flatten()).item()
    accuracy = accuracy_fun(pred.flatten(), targets.flatten()).item()
    batch_class_acc = multi_class_csi(pred, targets, confusion_metric)

    return jaccard, accuracy, batch_class_acc


def initiate_evaluation_dataloader(
        args: argparse.Namespace, batch_size: int
) -> Dict[str, DataLoader]:
    """
    Creates a dictionary of dataloaders depending on the args.evaluate dataset names.
    File stems are read from the supplied custom split json or created randomly.
    """
    preprocessing = create_preprocess(args)
    image_path, page_dataset, target_path = prepare_paths(args)

    if args.custom_split_file:

        with open(args.custom_split_file, "r", encoding="utf-8") as file:
            split = json.load(file)
            file_stems_dict = {}
            for dataset in args.evaluate:
                assert (
                        dataset in split.keys()
                ), f"'{dataset}' is not a valid key for the custom split dictionary!"
                file_stems_dict[dataset] = split[dataset]
    else:
        _, _, test_pages = page_dataset.random_split(args.split_ratio)
        file_stems_dict = {"Test": test_pages.file_stems}

    dataloader_dict = create_dataloader(
        args, batch_size, file_stems_dict, image_path, preprocessing, target_path
    )

    return dataloader_dict


def create_dataloader(
        args: argparse.Namespace,
        batch_size: int,
        file_stems_dict: Dict[str, List[str]],
        image_path: str,
        preprocessing: Preprocessing,
        target_path: str,
) -> Dict[str, DataLoader]:
    """
    Iterates over the file_stems_dict and creates a dataset with dataloader for each file stem list.
    All datasets that are not named "train" are considered validation/test datsets  and therefore no augmentations
    are applied, nor will the dataset be shuffled.
    """
    dataloader_dict = {}
    for key, file_stems in file_stems_dict.items():
        limit = args.limit if key == "Training" else None
        dataset = CropDataset(
            preprocessing,
            image_path=image_path,
            target_path=target_path,
            limit=limit,
            dataset=args.dataset,
            scale_aug=args.scale_aug,
            file_stems=file_stems,
            name=key,
        )
        print(f"{key} size: {len(dataset)}")
        if not key == "Training":
            dataset.augmentations = False
        dataloader_dict[key] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=key == "Training",
            num_workers=args.num_workers,
            drop_last=True,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=True,
            pin_memory=True,
        )
        assert len(dataloader_dict[key]) > 0, (
            f"Dataset {key} is to small to assemble at least one batch of "
            f"{batch_size} crops!"
        )
    return dataloader_dict


def initiate_dataloader(
        args: argparse.Namespace, batch_size: int
) -> Dict[str, DataLoader]:
    """
    Creates train, val and test datasets according to train.py args.
    """
    preprocessing = create_preprocess(args)
    image_path, page_dataset, target_path = prepare_paths(args)

    test_file_stems, train_file_stems, val_file_stems = get_file_stem_split(
        args.custom_split_file, args.split_ratio, page_dataset
    )

    file_stems_dict = {
        "Training": train_file_stems,
        "Validation": val_file_stems,
        "Test": test_file_stems,
    }
    dataloader_dict = create_dataloader(
        args, batch_size, file_stems_dict, image_path, preprocessing, target_path
    )

    return dataloader_dict


def prepare_paths(args: argparse.Namespace) -> Tuple[str, PageDataset, str]:
    "Adjust paths and create page_dataset."
    image_path = f"{adjust_path(args.data_path)}images/"
    target_path = f"{adjust_path(args.data_path)}targets/"
    page_dataset = PageDataset(Path(image_path), args.dataset)
    return image_path, page_dataset, target_path


def create_preprocess(args: argparse.Namespace) -> Preprocessing:
    "Initializes preprocessing Object"
    preprocessing = Preprocessing(
        scale=args.scale,
        crop_factor=args.crop_factor,
        crop_size=args.crop_size,
        reduce_classes=args.reduce_classes,
    )
    return preprocessing


def multi_class_csi(
        pred: torch.Tensor, target: torch.Tensor, metric: MulticlassConfusionMatrix
) -> torch.Tensor:
    """Calculate csi score using true positives, true negatives and false negatives from confusion matrix.
    Csi score is used as substitute for accuracy, calculated separately for each class.
    Returns numpy array with an entry for every class. If every prediction is a true negative,
    the score cant be calculated and the array will contain nan. These cases should be completely ignored.
    :param pred: prediction tensor
    :param target: target tensor
    :return:
    """
    pred = pred.flatten()
    target = target.flatten()

    matrix: torch.Tensor = metric(pred, target)
    true_positive = torch.diagonal(matrix)
    false_positive = torch.sum(matrix, dim=1) - true_positive
    false_negative = torch.sum(matrix, dim=0) - true_positive
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        csi = true_positive / (true_positive + false_negative + false_positive)
    return csi


def multi_precison_recall(
        pred: torch.Tensor, target: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate precision and recall using true positives, true negatives and false negatives from confusion matrix.
    Returns numpy array with an entry for every class. If every prediction is a true negative,
    the score cant be calculated and the array will contain nan. These cases should be completely ignored.
    :param pred: prediction tensor
    :param target: target tensor
    :return:
    """
    if pred.dim() == 4:
        pred = torch.argmax(pred, dim=1).type(torch.uint8)

    pred = pred.flatten()
    target = target.flatten()

    metric: MulticlassConfusionMatrix = MulticlassConfusionMatrix(
        num_classes=OUT_CHANNELS
    ).to(pred.device)

    # extract number of pixel for each class for pred and target and take the max value.
    target_counts = torch.zeros(OUT_CHANNELS, dtype=torch.long).to(target.device)
    unique_counts = torch.unique(target, return_counts=True)
    print(unique_counts)
    print(target.shape)
    target_counts[unique_counts[0].to(torch.long)] = unique_counts[1].to(torch.long)

    pred_counts = torch.zeros(OUT_CHANNELS, dtype=torch.long).to(pred.device)
    unique_counts = torch.unique(pred, return_counts=True)
    print(unique_counts)
    print(pred.shape)
    pred_counts[unique_counts[0].to(torch.long)] = unique_counts[1].to(torch.long)

    pixel_counts = torch.max(target_counts, pred_counts)

    # calaculate matrics
    # pylint: disable=not-callable
    matrix: torch.Tensor = metric(pred, target)
    true_positive = torch.diagonal(matrix)
    false_positive = torch.sum(matrix, dim=1) - true_positive
    false_negative = torch.sum(matrix, dim=0) - true_positive
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precision = torch.tensor(true_positive / (true_positive + false_positive))
        recall = torch.tensor(true_positive / (true_positive + false_negative))
        f1_score = torch.tensor(
            2 * true_positive / (2 * true_positive + false_negative + false_positive)
        )
    return precision, recall, f1_score, pixel_counts
