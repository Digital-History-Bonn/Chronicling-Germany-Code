from NewsDataset import NewsDataset
from model import dhSegment
from utils import RollingAverage

from sklearn.metrics import jaccard_score, accuracy_score
import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

EPOCHS = 1
BATCH_SIZE = 1
DATALOADER_WORKER = 4
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
    model = dhSegment([3, 4, 6, 4], in_channels=IN_CHANNELS, out_channel=OUT_CHANNELS, load_resnet_weights=True)
    model = model.float()

    # load model if argument is None it does nothing
    model.load(load_model)

    # load data
    dataset = NewsDataset()

    # splitting with fractions should work according to pytorch doc, but it does not
    train_set, test_set, _ = dataset.random_split([.9, .05, .05])
    print(f"train size: {len(train_set)}, test size: {len(test_set)}")

    print(f"ration between classes: {train_set.class_ratio(OUT_CHANNELS)}")

    # set optimizer and loss_fn
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = CrossEntropyLoss()                            # weight=torch.tensor(LOSS_WEIGHTS)

    # train
    for e in range(1, EPOCHS + 1):
        print(f"start epoch {e}:")
        train_loop(train_set, model, loss_fn, optimizer)

    model.save(save_model)

    # validation
    validation(test_set, model, loss_fn)


def train_loop(data: NewsDataset, model: torch.nn.Module, loss_fn, optimizer):
    """
    executes one trainings epoch
    :param data: Dataloader object with data to train on
    :param model: model to train
    :param loss_fn: loss function to optimize
    :param optimizer: optimizer to use
    :return: None
    """
    device = 'cpu' # "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    if device == 'cuda':
        model.cuda()

    size = len(data)
    data = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, num_workers=DATALOADER_WORKER)

    rolling_average = RollingAverage(10)
    t = tqdm(data, desc='train_loop', total=size)
    for X, Y, _ in t:
        X = X.to(device)
        Y = Y.to(device)

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, Y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update description
        t.set_description(f"loss: {rolling_average(loss.detach())}")


def validation(data: NewsDataset, model, loss_fn):
    """
    validation
    :param data: Dataloader with data to validate on
    :param model: model to validate
    :param loss_fn: loss_fn to validate with
    :return: None
    """
    size = len(data)
    data = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, num_workers=DATALOADER_WORKER)
    loss_sum = 0
    jaccard_sum = 0
    accuracy_sum = 0
    for batch, (X, Y, _) in tqdm(enumerate(data), desc='validation_loop', total=size):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, Y).detach().numpy()
        loss_sum += loss
        pred = np.argmax(pred.detach().numpy(), axis=1)
        jaccard_sum += jaccard_score(Y[0].flatten(), pred.flatten(), average='macro')
        accuracy_sum += accuracy_score(Y[0].flatten(), pred.flatten())

    print(f"average loss: {loss_sum / size}")
    print(f"average accuracy: {accuracy_sum / size}")
    print(f"average jaccard score: {jaccard_sum / size}")  # Intersection over Union


if __name__ == '__main__':
    train(load_model=None, save_model='Models/model.pt')
