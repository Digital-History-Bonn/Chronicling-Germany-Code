"""Module for utility function for training. This includes initialization of the model, dataset and dataloaders, as
well as computing metrics for validation and test."""
import argparse
import json
import warnings
from typing import Union, Any, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix

from src.news_seg.datasets.page_dataset import PageDataset
from src.news_seg.datasets.train_dataset import TrainDataset

from src.news_seg.models.dh_segment import DhSegment, conv1x1
from src.news_seg.models.dh_segment_cbam import DhSegmentCBAM
from src.news_seg.models.dh_segment_small import DhSegmentSmall
from src.news_seg.models.trans_unet import VisionTransformer
from src.news_seg.processing.preprocessing import Preprocessing

from src.news_seg.train_config import IN_CHANNELS, OUT_CHANNELS


def init_model(load: Union[str, None], device: str, model_str: str, freeze: bool = True,
               skip_cbam: bool = False, overwrite_load_channels: int = OUT_CHANNELS) -> Any:
    """
    Initialise model
    :param args:
    :param load: contains path to load the model from. If False, the model will be initialised randomly
    :param freeze: activates encoder freezing
    :param skip_cbam: activates cbam skip connection
    :return: loaded model
    """
    if model_str == "dh_segment":
        # create model
        model: Any = DhSegment(
            [3, 4, 6, 4],
            in_channels=IN_CHANNELS,
            out_channel=overwrite_load_channels,
            load_resnet_weights=True,
        )
        model = setup_dh_segment(device, load, model, freeze, overwrite_load_channels)
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
            in_channels=IN_CHANNELS, out_channel=OUT_CHANNELS, load_resnet_weights=True, cbam_skip_connection=skip_cbam
        )
        model = setup_dh_segment(device, load, model, freeze)
    elif model_str == "dh_segment_small":
        model = DhSegmentSmall(
            in_channels=IN_CHANNELS, out_channel=OUT_CHANNELS, load_resnet_weights=True
        )
        model = setup_dh_segment(device, load, model, freeze)
    assert model, "No valid model string supplied in model parameter"
    return model


def setup_dh_segment(
        device: str, load: Union[str, None], model: Any, freeze: bool,
        override_load_channels: int = OUT_CHANNELS) -> Any:
    """
    Setup function for dh_segment and dh_segment_cbam
    :param override_load_channels: overrides the out channel number with that a model will be loaded.
    :param freeze: freezes the encoder
    :param device:
    :param load: contains path to load the model from. If False, the model will be initialised randomly
    :param model:
    :return:
    """
    model = model.float()
    if freeze:
        model.freeze_encoder()
    # load model if argument is None, it does nothing
    model.load(load, device)
    if override_load_channels != OUT_CHANNELS:
        model.conv2 = conv1x1(32, 12)
        model.out_channel = 12
        print(
            f"overriding model loading out channels. {override_load_channels} channels are "
            f"loaded and overwritten with {OUT_CHANNELS} channels")
    # set mean and std in a model for normalization
    model.means = torch.tensor((0.485, 0.456, 0.406))
    model.stds = torch.tensor((0.229, 0.224, 0.225))
    return model


def load_score(load: Union[str, None], args: argparse.Namespace) -> Tuple[float, int, int]:
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
        gamma: float = 2.0
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
    probalbilites = torch.nn.functional.softmax(prediction, dim=1)
    focus = torch.pow(1 - probalbilites, gamma)
    loss = 1 - target * weights[None, :, None, None] * focus * probalbilites
    return torch.sum(loss, dim=1) / OUT_CHANNELS  # type: ignore


def calculate_scores(data: torch.Tensor) -> Tuple[float, float, Tensor]:
    """
    Applies argmax on the channel dimension of the prediction to obtain int class values.
    Then computes jaccard, accuracy and multi class csi score (critical sucess index)
    :param data: Combined and flattened prediction and Target with shape [P, C] P being number of pixels.
    The last Channel contains target data.
    """
    pred = data[:, : -1]
    targets = torch.squeeze(data[:, -1].to(torch.uint8))

    jaccard_fun = JaccardIndex(task="multiclass", num_classes=OUT_CHANNELS, average="weighted").to(
        pred.get_device())  # type: ignore
    accuracy_fun = MulticlassAccuracy(num_classes=OUT_CHANNELS, average="weighted").to(pred.get_device())
    confusion_metric = MulticlassConfusionMatrix(num_classes=OUT_CHANNELS).to(pred.get_device())

    pred = torch.argmax(pred, dim=1).type(torch.uint8)

    # pylint: disable=not-callable
    jaccard = jaccard_fun(pred.flatten(), targets.flatten()).item()
    accuracy = accuracy_fun(pred.flatten(), targets.flatten()).item()
    batch_class_acc = multi_class_csi(pred, targets, confusion_metric)

    return jaccard, accuracy, batch_class_acc


def initiate_datasets(args: argparse.Namespace) -> Tuple[TrainDataset, ...]:
    """
    Creates train, val and test datasets according to train.py args.
    """
    preprocessing = Preprocessing(
        scale=args.scale,
        crop_factor=args.crop_factor,
        crop_size=args.crop_size,
        reduce_classes=args.reduce_classes
    )
    image_path = f"{args.data_path}images/"
    target_path = f"{args.data_path}targets/"
    page_dataset = PageDataset(image_path, args.dataset)

    if args.custom_split_path:
        with open(args.custom_split_path + "custom-split.json", "r", encoding="utf-8") as file:
            split = json.load(file)
            train_file_stems = split[0]
            val_file_stems = split[1]
            test_file_stems = split[2]
            print(
                f"custom page level split with train size {len(train_file_stems)}, val size"
                f" {len(val_file_stems)} and test size {len(test_file_stems)}")
    else:
        train_pages, validation_pages, test_pages = page_dataset.random_split(args.split_ratio)
        train_file_stems = train_pages.file_stems
        val_file_stems = validation_pages.file_stems
        test_file_stems = test_pages.file_stems

        with open("custom-split.json", "w", encoding="utf8") as file:
            json.dump((train_file_stems, val_file_stems, test_file_stems), file)

    train_set = TrainDataset(
        preprocessing,
        image_path=image_path,
        target_path=target_path,
        limit=args.limit,
        dataset=args.dataset,
        scale_aug=args.scale_aug,
        file_stems=train_file_stems,
        name="train"
    )
    validation_set = TrainDataset(
        preprocessing,
        image_path=image_path,
        target_path=target_path,
        dataset=args.dataset,
        scale_aug=args.scale_aug,
        file_stems=val_file_stems,
        name="val"
    )
    test_set = TrainDataset(
        preprocessing,
        image_path=image_path,
        target_path=target_path,
        dataset=args.dataset,
        scale_aug=args.scale_aug,
        file_stems=test_file_stems,
        name="test"
    )
    print(f"train size: {len(train_set)}, test size: {len(validation_set)}")

    # Turn of augmentations on Validation-set
    validation_set.augmentations = False
    test_set.augmentations = False
    return train_set, validation_set, test_set


def initiate_dataloader(args: argparse.Namespace, batch_size: int, test_set: TrainDataset, train_set: TrainDataset,
                        validation_set: TrainDataset) -> Tuple[DataLoader, ...]:
    """
    Initiates train val and test dataloaders. Traindataloader is shuffled, while val and test are not.
    """
    # init dataloader
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        validation_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=True,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=True,
    )
    assert (
            len(train_loader) > 0
            and len(val_loader) > 0
            and len(test_loader) > 0
    ), "At least one Dataset is to small to assemble at least one batch"

    return train_loader, val_loader, test_loader


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
        pred: torch.Tensor, target: torch.Tensor, out_channels: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate precision and recall using true positives, true negatives and false negatives from confusion matrix.
    Returns numpy array with an entry for every class. If every prediction is a true negative,
    the score cant be calculated and the array will contain nan. These cases should be completely ignored.
    :param pred: prediction tensor
    :param target: target tensor
    :return:
    """

    pred = torch.argmax(pred, dim=1).type(torch.uint8)

    metric: MulticlassConfusionMatrix = MulticlassConfusionMatrix(num_classes=out_channels).to(pred.get_device())

    pred = pred.flatten()
    target = target.flatten()

    # pylint: disable=not-callable
    matrix: torch.Tensor = metric(pred, target)
    true_positive = torch.diagonal(matrix)
    false_positive = torch.sum(matrix, dim=1) - true_positive
    false_negative = torch.sum(matrix, dim=0) - true_positive
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precision = torch.tensor(
            true_positive / (true_positive + false_positive)
        )
        recall = torch.tensor(
            true_positive / (true_positive + false_negative)
        )
    return precision, recall
