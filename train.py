"""
module for training the hdSegment Model
"""

import datetime
import argparse
from typing import List, Union
import random

import numpy as np
from sklearn.metrics import jaccard_score, accuracy_score  # type: ignore
import tensorflow as tf  # type: ignore
import torch  # type: ignore
import tqdm  # type: ignore
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy  # type: ignore
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


class Trainer:
    def __init__(self, load_model: Union[str, None] = None, save_model: Union[str, None] = None,
                 batch_size: int = BATCH_SIZE, lr: float = LEARNING_RATE):
        """
        Trainer-class to train DhSegment Model
        :param load_model: model to load, init random if None
        :param save_model: name of the model in savefile and on tensorboard
        :param batch_size: size of batches
        :param lr: learning-rate
        """

        # init params
        self.save_model = save_model
        self.lr: float = lr
        self.batch_size: int = batch_size
        self.step: int = 0
        self.epoch: int = 0

        # create model
        self.model = DhSegment([3, 4, 6, 4], in_channels=IN_CHANNELS, out_channel=OUT_CHANNELS,
                               load_resnet_weights=True)
        self.model = self.model.float()
        self.model.freeze_encoder()

        # load model if argument is None it does nothing
        self.model.load(load_model)

        # set mean and std in model for normalization
        self.model.means = torch.tensor((0.485, 0.456, 0.406))
        self.model.stds = torch.tensor((0.229, 0.224, 0.225))

        # set optimizer and loss_fn
        self.optimizer = Adam(self.model.parameters(), lr=lr)  # weight_decay=1e-4
        self.loss_fn = CrossEntropyLoss(weight=torch.tensor(LOSS_WEIGHTS))

        # load data
        dataset = NewsDataset(scale=args.scale)

        # splitting with fractions should work according to pytorch doc, but it does not
        train_set, validation_set, _ = dataset.random_split((.9, .05, .05))
        print(f"train size: {len(train_set)}, test size: {len(validation_set)}")

        # init dataloader
        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                       num_workers=DATALOADER_WORKER, drop_last=True)
        self.val_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False,
                                     num_workers=DATALOADER_WORKER, drop_last=True)

        # check for cuda
        self.device = "cuda:1" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device} device")

    def train(self, epochs: int = 1):
        """
        executes all training epochs. After each epoch a validation round is performed.
        :param epochs: number of epochs that will be executed
        :return: None
        """

        self.model.to(self.device)
        self.loss_fn.to(self.device)

        self.step = 0
        for self.epoch in range(1, epochs + 1):
            self.model.train()

            with tqdm.tqdm(total=(len(self.train_loader)), desc=f'Epoch {self.epoch}/{epochs}', unit='image') as pbar:
                for images, targets in self.train_loader:
                    # because of cropping, each element in a batch contains all cropped images of one full size image. This
                    # tensor will be flattend to be shuffled within the batch
                    data = torch.cat((torch.flatten(images, start_dim=0, end_dim=1),
                                      torch.flatten(targets, start_dim=0, end_dim=1)[:, np.newaxis, :, :]), dim=1)
                    indices = torch.randperm(data.shape[0])
                    loss = self.run_batches(data[indices])

                    # update description
                    pbar.update(1)
                    pbar.set_postfix(**{'loss (batch)': loss})

                    # delete data from gpu cache
                    del images, targets, loss, data
                    torch.cuda.empty_cache()

            # save model at end of epoch
            self.model.save(self.save_model)

    def run_batches(self, data: torch.Tensor) -> float:
        """Runs training batch-wise.
        :param data: Contains cropped and shuffled images and targets. These are selected from a number of
        original images, according to BATCH_SIZE
        :return: average loss of the batch
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
            augmentations = self.get_augmentations()
            batch_data = augmentations(batch_data)
            batch_images = batch_data[:, :-1].to(device=self.device, dtype=torch.float32)
            batch_targets = batch_data[:, -1].to(device=self.device, dtype=torch.long)

            preds = self.model(batch_images)
            batch_loss = self.loss_fn(preds, batch_targets)

            # Backpropagation
            self.optimizer.zero_grad(set_to_none=True)
            batch_loss.backward()
            self.optimizer.step()
            loss += batch_loss.item()

            # update tensor board logs
            self.step += 1
            with summary_writer.as_default():
                tf.summary.scalar('train loss', batch_loss.item(), step=self.step)
                tf.summary.scalar('batch mean', batch_images.detach().cpu().mean(), step=self.step)
                tf.summary.scalar('batch std', batch_images.detach().cpu().std(), step=self.step)
                tf.summary.scalar('target batch mean', batch_targets.detach().cpu().float().mean(), step=self.step)
                tf.summary.scalar('target batch std', batch_targets.detach().cpu().float().std(), step=self.step)

            if self.step % VAL_EVERY == 0:
                self.validation()

            del batch_images, batch_targets, batch_loss, batch_data, preds
            torch.cuda.empty_cache()

        return loss / count

    def validation(self):
        """
        Executes one validation round, containing the evaluation of the current model on the entire validation set.
        :return: None
        """
        self.model.eval()
        size = len(self.val_loader)
        count = 0

        loss_sum, jaccard_sum, accuracy_sum, class_acc_sum = 0, 0, 0, np.zeros(OUT_CHANNELS)

        for images, targets in tqdm.tqdm(self.val_loader, desc='validation_round', total=size, unit='image'):
            # Compute prediction and loss
            data = torch.cat((torch.flatten(images, start_dim=0, end_dim=1),
                              torch.flatten(targets, start_dim=0, end_dim=1)[:, np.newaxis, :, :]), dim=1)
            count += data.shape[0]
            batch_loss, batch_jaccard_sum, batch_accuracy_sum, batch_class_acc_sum = self.run_val_batches(data)
            loss_sum += batch_loss
            jaccard_sum += batch_jaccard_sum
            accuracy_sum += batch_accuracy_sum
            class_acc_sum += batch_class_acc_sum

        self.val_logging(loss_sum / count, jaccard_sum / count,
                         accuracy_sum / (count * 3 * 255 * 255),
                         class_acc_sum / (count * 3 * 255 * 255))

    def run_val_batches(self, data):
        """
        Executes validation batches.
        :param data: Contains cropped and shuffled images and targets. These are selected from a number of
        original images, according to batch_size
        """
        jaccard_sum = 0
        accuracy_sum = 0
        loss = 0.0
        multi_class_accuracy = MulticlassAccuracy(num_classes=OUT_CHANNELS)
        class_acc_sum = np.zeros(OUT_CHANNELS)

        # second for loop related batchsize
        batch_size = 32
        for batch_data in torch.split(data, batch_size):
            batch_images = batch_data[:, :-1].to(device=self.device, dtype=torch.float32)
            batch_targets = batch_data[:, -1].to(device=self.device, dtype=torch.long)

            pred = self.model(batch_images)
            batch_loss = self.loss_fn(pred, batch_targets)

            # detach results
            pred = pred.detach().cpu().numpy()
            batch_loss = batch_loss.detach().cpu().numpy()

            batch_targets = batch_targets.detach().cpu().numpy()

            loss += batch_loss
            pred = np.argmax(pred, axis=1)
            jaccard_sum += jaccard_score(batch_targets.flatten(), pred.flatten(), average='micro')
            accuracy_sum += accuracy_score(batch_targets.flatten(), pred.flatten(), normalize=False)
            class_acc_sum += multi_class_accuracy(torch.tensor(pred).flatten(),
                                                  torch.tensor(batch_targets).flatten()).numpy()

            del pred, batch_loss
            torch.cuda.empty_cache()

            return loss, jaccard_sum, accuracy_sum, class_acc_sum

    def val_logging(self, loss, jaccard, accuracy, class_accs):
        """Handles logging for loss values and validation images. Per epoch one random cropped image from the
        validation set will be evaluated. Furthermore, one full size image will be predicted and logged.
        :param loss: loss sum for validation round
        :param jaccard: jaccard score of validation round
        :param accuracy: accuracy of validation round
        :param class_accs: array of accuracy by class
        """
        size = len(self.val_loader)
        image, target = self.val_loader.dataset[
            random.randint(0, (size * self.val_loader.batch_size if self.val_loader.batch_size else 1) - 1)]
        rand_index = random.randint(0, image.shape[0])
        image = torch.unsqueeze(image[rand_index].to(self.device), 0)
        pred = self.model(image).argmax(dim=1).float()
        log_image = get_file(PREDICT_IMAGE, PREDICT_SCALE)
        log_pred = self.model.predict(log_image.to(self.device))

        # update tensor board logs
        with summary_writer.as_default():
            tf.summary.scalar('epoch', self.epoch, step=self.step)

            tf.summary.scalar('val loss', loss, step=self.step)
            tf.summary.scalar('val accuracy', accuracy, step=self.step)

            for i in range(OUT_CHANNELS):
                tf.summary.scalar(f'val accuracy for class {i}', class_accs[i], step=self.step)

            tf.summary.scalar('val jaccard score', jaccard, step=self.step)

            for i, acc in enumerate(class_accs):
                tf.summary.scalar(f'val accuracy for class {i}', acc, step=self.step)

            tf.summary.image('val image', torch.permute(image.cpu()/255, (0, 2, 3, 1)), step=self.step)
            tf.summary.image('val target', target[rand_index].float().cpu()[None, :, :, None] / OUT_CHANNELS,
                             step=self.step)
            tf.summary.image('val prediction', pred.float().cpu()[:, :, :, None] / OUT_CHANNELS, step=self.step)

            tf.summary.image('full site prediction input', torch.permute(log_image.cpu()/255, (0, 2, 3, 1)), step=self.step)
            tf.summary.image('full site prediction result', log_pred[None, :, :, None], step=self.step)

        print(f"average loss: {loss}")
        print(f"average accuracy: {accuracy}")
        print(f"average jaccard score: {jaccard}")  # Intersection over Union

        del size, image, target, pred, log_image, log_pred
        torch.cuda.empty_cache()

    @staticmethod
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
    PREDICT_SCALE = args.predict_scale
    PREDICT_IMAGE = args.predict_image

    # setup tensor board
    train_log_dir = 'logs/runs/' + args.name
    summary_writer = tf.summary.create_file_writer(train_log_dir)

    load_model = f'Models/model_{args.load}.pt' if args.load else None

    trainer = Trainer(load_model=load_model, save_model=f'Models/model_{args.name}.pt',
                      batch_size=args.batch_size, lr=args.lr)

    trainer.train(epochs=args.epochs)
