"""
module for training the hdSegment Model
"""

import datetime
import argparse

import numpy as np
import sklearn.metrics  # type: ignore
import tensorflow as tf  # type: ignore
import torch  # type: ignore
import tqdm  # type: ignore
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms  # type: ignore

from model import DhSegment
from news_dataset import NewsDataset
from predict import get_file

EPOCHS = 1
VAL_EVERY = 250
BATCH_SIZE = 32
DATALOADER_WORKER = 4
IN_CHANNELS, OUT_CHANNELS = 3, 10
LEARNING_RATE = .001  # 0,0001 seems to work well
LOSS_WEIGHTS = [1.0, 10.0, 10.0, 10.0, 1.0, 10.0, 10.0, 10.0, 10.0, 10.0]  # 1 and 5 seems to work well

LOGGING_IMAGE = "../prima/inputs/NoAnnotations/00675238.tif"

# set random seed for reproducibility
torch.manual_seed(42)


def train(load_model=None, save_model=None, epochs: int = EPOCHS):
    """
    train function. Initializes dataloaders and optimzer.
    :param load_model: (default: None) path to model to load
    :param save_model: (default: None) path to save the model
    :param epochs: (default: EPOCHS) number of epochs
    :return: None
    """
    # create model
    model = DhSegment([3, 4, 6, 4], in_channels=IN_CHANNELS, out_channel=OUT_CHANNELS, load_resnet_weights=True)

    model = model.float()

    # load model if argument is None it does nothing
    model.load(load_model)

    # load data
    dataset = NewsDataset()

    # splitting with fractions should work according to pytorch doc, but it does not
    train_set, validation_set, _ = dataset.random_split((.9, .05, .05))
    print(f"train size: {len(train_set)}, test size: {len(validation_set)}")

    print(f"ration between classes: {train_set.class_ratio(OUT_CHANNELS)}")

    # set mean and std in model for normalization
    model.means = train_set.mean
    model.stds = train_set.std

    # set optimizer and loss_fn
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)  # weight_decay=1e-4
    loss_fn = CrossEntropyLoss(weight=torch.tensor(LOSS_WEIGHTS))  # weight=torch.tensor(LOSS_WEIGHTS)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                                               num_workers=DATALOADER_WORKER, drop_last=True)
    val_loader = DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=DATALOADER_WORKER)

    train_loop(train_loader, model, loss_fn, epochs, optimizer, val_loader)

    model.save(save_model)


def train_loop(train_loader: DataLoader, model: DhSegment, loss_fn: torch.nn.Module, epochs: int,
               optimizer: torch.optim.Optimizer, val_loader: DataLoader):
    """
    executes all training epochs. After each epoch a validation round is performed.
    :param train_loader: Dataloader object
    :param model: model to train
    :param loss_fn: loss function to optimize
    :param optimizer: optimizer to use
    :param val_loader: dataloader with validation data
    :param epochs: number of epochs that will be executed
    :return: None
    """

    model.to(DEVICE)
    loss_fn.to(DEVICE)

    step = 0
    for epoch in range(1, epochs + 1):
        model.train()

        with tqdm.tqdm(total=(len(train_loader)), desc=f'Epoch {epoch}/{epochs}', unit='batches') as pbar:
            for images, targets in train_loader:
                images = images.to(DEVICE)
                targets = targets.to(DEVICE)

                # Compute prediction and loss
                pred = model(images)
                loss = loss_fn(pred, targets)

                # Backpropagation
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                # update description
                pbar.update(1)
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # update tensor board logs
                step += 1
                with summary_writer.as_default():
                    tf.summary.scalar('train loss', loss.item(), step=step)

                # delete data from gpu cache
                del images, targets, pred, loss
                torch.cuda.empty_cache()

                if step % VAL_EVERY == 0:
                    validation(val_loader, model, loss_fn, epoch, step)


def validation(val_loader: DataLoader, model, loss_fn, epoch: int, step: int):
    """
    Executes one validation round, containing the evaluation of the current model on the entire validation set.
    :param model: model to validate
    :param loss_fn: loss_fn to validate with
    :param val_loader: dataloader with validation data
    :param epoch: current epoch value for logging
    :param step: current batch related step value for logging. Count of batches that have been loaded.
    :return: None
    """

    model.eval()

    size = len(val_loader)

    loss_sum = 0
    jaccard_sum = 0
    accuracy_sum = 0
    for images, targets in tqdm.tqdm(val_loader, desc='validation_round', total=size):
        # Compute prediction and loss
        augmentations = get_augmentations()
        data = augmentations(torch.concat((images, targets[:, np.newaxis, :, :]), dim=1))
        images = data[:, :-1].to(device=DEVICE, dtype=torch.float32)
        targets = data[:, -1].to(device=DEVICE, dtype=torch.long)

        pred = model(images)
        loss = loss_fn(pred, targets)

        pred = pred.detach().cpu().numpy()
        loss = loss.detach().cpu().numpy()

        targets = targets.detach().cpu().numpy()

        loss_sum += loss
        pred = np.argmax(pred, axis=1)
        jaccard_sum += sklearn.metrics.jaccard_score(targets.flatten(), pred.flatten(), average='macro')
        accuracy_sum += sklearn.metrics.accuracy_score(targets.flatten(), pred.flatten())

        del images, targets, pred, loss
        torch.cuda.empty_cache()

    image, target = val_loader.dataset[0]
    image = torch.unsqueeze(image.to(DEVICE), 0)
    pred = model(image).argmax(dim=1).float()

    # update tensor board logs
    with summary_writer.as_default():
        tf.summary.scalar('val loss', loss_sum / size, step=step)
        tf.summary.scalar('val accuracy', accuracy_sum / size, step=step)
        tf.summary.scalar('val jaccard score', jaccard_sum / size, step=step)
        tf.summary.scalar('epoch', epoch, step=step)
        tf.summary.image('val image', torch.transpose(image.cpu(), 3, 1),
                         step=step)
        tf.summary.image('val target', torch.unsqueeze(
            torch.unsqueeze(target.float().cpu() / OUT_CHANNELS, 0), 3), step=step)
        tf.summary.image('val prediction', torch.unsqueeze(pred.float().cpu() / OUT_CHANNELS, 3), step=step)
        tf.summary.image('full site prediction', log_pred(model)[None, :, :, None].repeat(1, 1, 1, 3), step=step)

    print(f"average loss: {loss_sum / size}")
    print(f"average accuracy: {accuracy_sum / size}")
    print(f"average jaccard score: {jaccard_sum / size}")  # Intersection over Union

    del image, target, pred
    torch.cuda.empty_cache()


def log_pred(model: DhSegment) -> torch.Tensor:
    """calls load and predict function for full image prediction
    :param model: prediction model"""
    image = get_file(LOGGING_IMAGE)
    pred = model.predict(image.to(DEVICE))
    return pred


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
    parser.add_argument('--epochs', '-e', metavar='EPOCHS', type=int, default=1, help='number of epochs to train')
    parser.add_argument('--name', '-n', metavar='NAME', type=str,
                        default=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                        help='name of run in tensorboard')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using {DEVICE} device")

    # setup tensor board
    train_log_dir = 'logs/runs/' + args.name
    summary_writer = tf.summary.create_file_writer(train_log_dir)

    train(load_model=None, save_model='Models/model.pt', epochs=args.epochs)
