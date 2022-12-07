"""
module for training the hdSegment Model
"""

import datetime
import argparse
from typing import List, Tuple
import random

import numpy as np
import sklearn.metrics  # type: ignore
import tensorflow as tf  # type: ignore
import torch  # type: ignore
import tqdm  # type: ignore
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy # type: ignore
from torchvision import transforms  # type: ignore
from torchmetrics.classification import MulticlassAccuracy  # type: ignore

import preprocessing
from model import DhSegment
from news_dataset import NewsDataset
from utils import get_file

EPOCHS = 1
VAL_EVERY = 250
BATCH_SIZE = 32
DATALOADER_WORKER = 1
IN_CHANNELS, OUT_CHANNELS = 3, 10
LEARNING_RATE = .001  # 0,0001 seems to work well
LOSS_WEIGHTS: List[float] = [1.0, 10.0, 10.0, 10.0, 1.0, 10.0, 10.0, 10.0, 10.0, 10.0]  # 1 and 5 seems to work well

PREDICT_SCALE = 0.25
PREDICT_IMAGE = "../prima/inputs/NoAnnotations/00675238.tif"

# set random seed for reproducibility
torch.manual_seed(42)


def train(load_model=None, save_model=None):
    """
    train function. Initializes dataloaders and optimzer.
    :param load_model: (default: None) path to model to load
    :param save_model: (default: None) path to save the model
    :return: None
    """

    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    # create model
    model = DhSegment([3, 4, 6, 4], in_channels=IN_CHANNELS, out_channel=OUT_CHANNELS, load_resnet_weights=True)

    model = model.float()
    model.freeze_encoder()

    # load model if argument is None it does nothing
    model.load(load_model)

    # load data
    dataset = NewsDataset(scale=args.scale)

    # splitting with fractions should work according to pytorch doc, but it does not
    train_set, validation_set, _ = dataset.random_split((.9, .05, .05))
    print(f"train size: {len(train_set)}, test size: {len(validation_set)}")

    # print(f"ration between classes: {train_set.class_ratio(OUT_CHANNELS)}")

    # set mean and std in model for normalization
    model.means = torch.tensor((0.485, 0.456, 0.406))
    model.stds = torch.tensor((0.229, 0.224, 0.225))

    # set optimizer and loss_fn
    optimizer = Adam(model.parameters(), lr=lr)  # weight_decay=1e-4

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               num_workers=DATALOADER_WORKER, drop_last=True)
    val_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=DATALOADER_WORKER,
                            drop_last=True)

    train_loop(train_loader, model, epochs, optimizer, val_loader, save_model, batch_size)


def train_loop(train_loader: DataLoader, model: DhSegment, epochs: int,
               optimizer: torch.optim.Optimizer, val_loader: DataLoader, save_model: str, batch_size: int):
    """
    executes all training epochs. After each epoch a validation round is performed.
    :param save_model: path for saving model
    :param train_loader: Dataloader object
    :param model: model to train
    :param optimizer: optimizer to use
    :param val_loader: dataloader with validation data
    :param epochs: number of epochs that will be executed
    :param batch_size: batch size
    :return: None
    """
    loss_fn = CrossEntropyLoss(weight=torch.tensor(LOSS_WEIGHTS))

    model.to(DEVICE)
    loss_fn.to(DEVICE)

    step = 0
    for epoch in range(1, epochs + 1):
        model.train()

        with tqdm.tqdm(total=(len(train_loader)), desc=f'Epoch {epoch}/{epochs}', unit='batches') as pbar:
            for images, targets in train_loader:
                # because of cropping, each element in a batch contains all cropped images of one full size image. This
                # tensor will be flattend to be shuffled within the batch
                data = torch.cat((torch.flatten(images, start_dim=0, end_dim=1),
                                  torch.flatten(targets, start_dim=0, end_dim=1)[:, np.newaxis, :, :]), dim=1)
                indices = torch.randperm(data.shape[0])
                data = data[indices]
                loss = run_batches(data, loss_fn, step, model, optimizer, batch_size, val_loader, epoch)

                # update description
                pbar.update(1)
                pbar.set_postfix(**{'loss (batch)': loss})

                # delete data from gpu cache
                del images, targets, loss, data
                torch.cuda.empty_cache()


        model.save(save_model)


def run_batches(data: torch.Tensor, loss_fn: torch.nn.Module, step: int, model: DhSegment, optimizer: torch.optim.Optimizer,
                batch_size: int, val_loader: DataLoader, epoch: int) -> float:
    """Runs training batch-wise.
    :param data: Contains cropped and shuffled images and targets. These are selected from a number of original images,
        according to BATCH_SIZE
    :param model: model to train
    :param loss_fn: loss function to optimize
    :param optimizer: optimizer to use
    :param batch_size: batch size
    :return:
    """
    count = 0
    loss = 0
    size = data.shape[0]
    # second for loop related batchsize
    batch_size = 32
    for i in range(0, size, int(batch_size)):
        if size - i < batch_size:
            break
        count += 1
        batch_data = data[i: i + batch_size]

        # Compute prediction and loss
        augmentations = get_augmentations()
        batch_data = augmentations(batch_data)
        batch_images = batch_data[:, :-1].to(device=DEVICE, dtype=torch.float32)
        batch_targets = batch_data[:, -1].to(device=DEVICE, dtype=torch.long)

        preds = model(batch_images)
        batch_loss = loss_fn(preds, batch_targets)

        # Backpropagation
        optimizer.zero_grad(set_to_none=True)
        batch_loss.backward()
        optimizer.step()
        loss += batch_loss.item()

        # update tensor board logs
        step += 1
        with summary_writer.as_default():
            tf.summary.scalar('train loss', batch_loss.item(), step=step)
            tf.summary.scalar('batch mean', batch_images.detach().cpu().mean(), step=step)
            tf.summary.scalar('batch std', batch_images.detach().cpu().std(), step=step)

        if step % VAL_EVERY == 0:
            validation(val_loader, model, loss_fn, epoch, step, batch_size)

        del batch_images, batch_targets, batch_loss, batch_data, preds
        torch.cuda.empty_cache()

    return loss / count


def validation(val_loader: DataLoader, model, loss_fn, epoch: int, step: int, batch_size: int):
    """
    Executes one validation round, containing the evaluation of the current model on the entire validation set.
    :param val_loader: dataloader with validation data
    :param model: model to validate
    :param loss_fn: loss_fn to validate with
    :param epoch: current epoch value for logging
    :param step: current batch related step value for logging. Count of batches that have been loaded.
    :param batch_size: batch size
    :return: None
    """
    model.eval()
    size = len(val_loader)

    for images, targets in tqdm.tqdm(val_loader, desc='validation_round', total=size):
        # Compute prediction and loss
        data = torch.cat((torch.flatten(images, start_dim=0, end_dim=1),
                          torch.flatten(targets, start_dim=0, end_dim=1)[:, np.newaxis, :, :]), dim=1)

        run_val_batches(batch_size, data, loss_fn, epoch, step, model, val_loader)


def run_val_batches(batch_size, data, loss_fn, epoch, step, model, val_loader):
    """
    Executes validation batches.
    :param batch_size: batch size
    :param data: Contains cropped and shuffled images and targets. These are selected from a number of original images,
        according to batch_size
    :param loss_fn: loss_fn to validate with
    :param epoch: current epoch value for logging
    :param step: current batch related step value for logging. Count of batches that have been loaded.
    :param model: model to validate
    :param val_loader: dataloader with validation data
    """
    jaccard_sum = 0
    accuracy_sum = 0
    count = 0
    loss = 0.0
    size = data.shape[0]
    multi_class_accuracy = MulticlassAccuracy(num_classes=OUT_CHANNELS)
    class_accs = np.zeros(OUT_CHANNELS)
    # second for loop related batchsize
    batch_size = 32
    for i in range(0, size, int(batch_size)):
        if size - i < batch_size:
            break
        count += 1
        batch_data = data[i: i + batch_size]
        count += 1
        batch_images = batch_data[:, :-1].to(device=DEVICE, dtype=torch.float32)
        batch_targets = batch_data[:, -1].to(device=DEVICE, dtype=torch.long)

        pred = model(batch_images)
        batch_loss = loss_fn(pred, batch_targets)

        pred = pred.detach().cpu().numpy()
        batch_loss = batch_loss.detach().cpu().numpy()

        batch_targets = batch_targets.detach().cpu().numpy()

        loss += batch_loss
        pred = np.argmax(pred, axis=1)
        jaccard_sum += sklearn.metrics.jaccard_score(batch_targets.flatten(), pred.flatten(), average='macro')
        accuracy_sum += sklearn.metrics.accuracy_score(batch_targets.flatten(), pred.flatten())
        class_accs += multi_class_accuracy(torch.tensor(pred).flatten(), torch.tensor(batch_targets).flatten()).numpy()

        del pred, batch_loss
        torch.cuda.empty_cache()

    val_logging(accuracy_sum, epoch, jaccard_sum, loss, class_accs / count, model, step, val_loader)


def val_logging(accuracy_sum, epoch, jaccard_sum, loss_sum, class_accs, model, step, val_loader):
    """Handles logging for loss values and validation images. Per epoch one random cropped image from the validation set
    will be evaluated. Furthermore, one full size image will be predicted and logged.

    :param accuracy_sum: accuracy sum for validation round
    :param epoch: current training epoch
    :param jaccard_sum: jaccard sum for validation round
    :param loss_sum: loss sum for validation round
    :param model: model to validate
    :param step: current training batch related step value for logging. Count of batches that have been loaded.
    :param val_loader: dataloader with validation data
    """
    size = len(val_loader)
    image, target = val_loader.dataset[random.randint(0, (size * val_loader.batch_size if val_loader.batch_size else 1) -1)]
    rand_index = random.randint(0, image.shape[0])
    image = torch.unsqueeze(image[rand_index].to(DEVICE), 0)
    pred = model(image).argmax(dim=1).float()
    log_image = get_file(predict_image, predict_scale)
    log_pred = model.predict(log_image.to(DEVICE))

    # update tensor board logs
    with summary_writer.as_default():
        tf.summary.scalar('epoch', epoch, step=step)

        tf.summary.scalar('val loss', loss_sum / size, step=step)
        tf.summary.scalar('val accuracy', accuracy_sum / size, step=step)

        for i in range(OUT_CHANNELS):
            tf.summary.scalar(f'val accuracy for class {i}', class_accs[i]/ size, step=step)

        tf.summary.scalar('val jaccard score', jaccard_sum / size, step=step)

        for i, acc in enumerate(class_accs):
            tf.summary.scalar(f'val accuracy for class {i}', acc / size, step=step)

        tf.summary.image('val image', torch.permute(image.cpu(), (0, 2, 3, 1)),
                         step=step)
        tf.summary.image('val target', target[rand_index].float().cpu()[None, :, :, None] / OUT_CHANNELS, step=step)
        tf.summary.image('val prediction', pred.float().cpu()[:, :, :, None] / OUT_CHANNELS, step=step)

        tf.summary.image('full site prediction input', torch.permute(log_image.cpu(), (0, 2, 3, 1)), step=step)
        tf.summary.image('full site prediction result', log_pred[None, :, :, None], step=step)

    print(f"average loss: {loss_sum / size}")
    print(f"average accuracy: {accuracy_sum / size}")
    print(f"average jaccard score: {jaccard_sum / size}")  # Intersection over Union

    del size, image, target, pred, log_image, log_pred
    torch.cuda.empty_cache()


def get_augmentations() -> transforms.Compose:
    """Defines transformations"""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180)
    ])


def get_args() -> argparse.Namespace:
    """defines arguments"""
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--epochs', '-e', metavar='EPOCHS', type=int, default=EPOCHS, help='number of epochs to train')
    parser.add_argument('--name', '-n', metavar='NAME', type=str,
                        default=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                        help='name of run in tensorboard')
    parser.add_argument('--predict_image', '-i', type=str,
                        default=PREDICT_IMAGE,
                        help='path for full image prediction')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('--learning-rate', '-lr', metavar='LR', type=float, default=LEARNING_RATE,
                        help='Learning rate', dest='lr')
    parser.add_argument('--scale', '-s', type=float, dest='scale', default=preprocessing.SCALE,
                        help='Downscaling factor of the images')
    parser.add_argument('--load', '-l', type=str, dest='load', default=None,
                        help='model to load (default is None)')
    parser.add_argument('--predict-scale', '-p', type=float, default=PREDICT_SCALE,
                        help='Downscaling factor of the predict image')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    predict_scale = args.predict_scale
    predict_image = args.predict_image

    DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(f"Using {DEVICE} device")

    # setup tensor board
    train_log_dir = 'logs/runs/' + args.name
    summary_writer = tf.summary.create_file_writer(train_log_dir)

    load_model = f'Models/model_{args.load}.pt' if args.load else None

    train(load_model=load_model, save_model=f'Models/model_{args.name}.pt')
