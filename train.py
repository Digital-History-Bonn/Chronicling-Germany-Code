from NewsDataset import NewsDataset
from model import DhSegment
from utils import RollingAverage

from sklearn.metrics import jaccard_score, accuracy_score
import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

EPOCHS = 5
BATCH_SIZE = 1
DATALOADER_WORKER = 1
IN_CHANNELS, OUT_CHANNELS = 1, 10
LEARNING_RATE = 0.0001  # 0,0001 seems to work well
# LOSS_WEIGHTS = [1.0, 9.0]  # 1 and 5 seems to work well

# set random seed for reproducibility
torch.manual_seed(42)


def train(load_model=None, save_model=None):
    """
    trainingsfunction
    :param load_model: (default: None) path to model to load
    :param save_model: (default: None) path to save the model
    :return: None
    """

    # create model
    model = DhSegment([3, 3, 4, 3], in_channels=IN_CHANNELS, out_channel=OUT_CHANNELS, load_resnet_weights=True)
    # [3, 4, 6, 4]
    model = model.float()

    # load model if argument is None it does nothing
    model.load(load_model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    if device == 'cuda':
        model.cuda()

    # load data
    dataset = NewsDataset()

    # splitting with fractions should work according to pytorch doc, but it does not
    train_set, test_set, _ = dataset.random_split([.9, .05, .05])
    print(f"train size: {len(train_set)}, test size: {len(test_set)}")

    print(f"ration between classes: {train_set.class_ratio(OUT_CHANNELS)}")

    # set optimizer and loss_fn
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = CrossEntropyLoss()  # weight=torch.tensor(LOSS_WEIGHTS)

    # train
    for e in range(1, EPOCHS + 1):
        print(f"start epoch {e}:")
        train_loop(train_set, model, loss_fn, optimizer)

    model.save(save_model)

    # validation
    validation(test_set, model, loss_fn)


def train_loop(data: NewsDataset, model: torch.nn.Module, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    executes one trainings epoch
    :param data: Dataloader object with data to train on
    :param model: model to train
    :param loss_fn: loss function to optimize
    :param optimizer: optimizer to use
    :return: None
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    size = len(data)
    data = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, num_workers=DATALOADER_WORKER)

    rolling_average = RollingAverage(10)
    t = tqdm(data, desc='train_loop', total=size)
    for images, masks, _ in t:
        # print(f"{X.shape=}")
        # print(f"start: {torch.cuda.memory_reserved()=}")
        images = images.to(device)
        masks = masks.to(device)

        # Compute prediction and loss
        pred = model(images)
        # print(f"pred: {torch.cuda.memory_reserved()=}")
        loss = loss_fn(pred, masks)
        # print(f"loss: {torch.cuda.memory_reserved()=}")
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        # print(f"backward: {torch.cuda.memory_reserved()=}")
        optimizer.step()
        # print(f"optimizer: {torch.cuda.memory_reserved()=}")
        # update description
        t.set_description(f"loss: {rolling_average(loss.detach())}")
        del images, masks, pred, loss
        torch.cuda.empty_cache()
        # print(f"end: {torch.cuda.memory_reserved()=}")
        # print()

        del images, masks, pred, loss
        torch.cuda.empty_cache()


def validation(data: NewsDataset, model, loss_fn):
    """
    validation
    :param data: Dataloader with data to validate on
    :param model: model to validate
    :param loss_fn: loss_fn to validate with
    :return: None
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    size = len(data)
    data = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, num_workers=DATALOADER_WORKER)
    loss_sum = 0
    jaccard_sum = 0
    accuracy_sum = 0
    for images, masks, _ in tqdm(data, desc='validation_loop', total=size):
        # Compute prediction and loss
        images = images.to(device)
        masks = masks.to(device)

        pred = model(images)
        loss = loss_fn(pred, masks)

        pred = pred.detach().cpu().numpy()
        loss = loss.detach().cpu().numpy()
        # images = images.detach().cpu().numpy()
        masks = masks.detach().cpu().numpy()

        loss_sum += loss
        pred = np.argmax(pred, axis=1)
        jaccard_sum += jaccard_score(masks.flatten(), pred.flatten(), average='macro')
        accuracy_sum += accuracy_score(masks.flatten(), pred.flatten())

        del images, masks, pred, loss
        torch.cuda.empty_cache()

    print(f"average loss: {loss_sum / size}")
    print(f"average accuracy: {accuracy_sum / size}")
    print(f"average jaccard score: {jaccard_sum / size}")  # Intersection over Union


if __name__ == '__main__':
    train(load_model=None, save_model='Models/model.pt')
