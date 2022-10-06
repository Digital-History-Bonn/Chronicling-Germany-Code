import torch
from torch.optim import AdamW, Adam
from torch.nn import CrossEntropyLoss, BCELoss
import torch.nn.functional as F

from tqdm import tqdm

from Dataloader import Dataloader
from model import dhSegment
from utils import plot_sampels, RollingAverage


EPOCHS = 10
BATCH_SIZE = 1
DATALOADER_WORKER = 4
IN_CHANNELS, OUT_CHANNELS = 1, 2
LEARNING_RATE = 0.0001                  # 0,0001 seems to work well
LOSS_WEIGHTS = [1.0, 9.0]               # 1 and 5 seems to work well

# TODO: add validationfunktion for baseline
# TODO: add comments


def train(train_data, validation_data=None, train_size=150, validation_size=50,
          sample_plot=3, load_model=None, save_model=None):
    """
    trainingsfunction
    :param train_data: path to trainings-data
    :param validation_data: path to validation-data
    :param train_size: limit for loaded trainings images
    :param validation_size: limit for loaded validation images
    :param sample_plot: number of images used for sample plot
    :param load_model: (default: None) path to model to load
    :param save_model: (default: None) path to save the model
    :return: None
    """

    #create model
    model = dhSegment([3, 4, 6, 4], in_channels=IN_CHANNELS, out_channel=OUT_CHANNELS, load_resnet_weights=True)
    model = model.float()

    model.load(load_model)

    # load train data
    train_dataloader = Dataloader(train_data, limit=train_size)
    print(f"ration between classes: {train_dataloader.class_ratio}")

    plot_sampels(train_dataloader, n=sample_plot, model=model, title='before training')

    # set optimizer and loss_fn
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = CrossEntropyLoss(weight=torch.tensor(LOSS_WEIGHTS))

    # train
    for e in range(1, EPOCHS+1):
        train_loop(train_dataloader, model, loss_fn, optimizer)
        plot_sampels(train_dataloader, n=2, model=model, title=f'after epoch: {e}')

    model.save(save_model)

    if validation_data is None:
        return

    # load validation_data
    test_dataloader = Dataloader(validation_data, limit=validation_size)

    # validation
    validation(test_dataloader, model, loss_fn)

    plot_sampels(test_dataloader, n=sample_plot, model=model, title=f'Samples from validation after training')


def train_loop(dataloader, model, loss_fn, optimizer):
    """
    executes one trainings epoch
    :param dataloader: Dataloader object with data to train on
    :param model: model to train
    :param loss_fn: loss function to optimize
    :param optimizer: optimizer to use
    :return: None
    """
    dataloader = torch.utils.data.DataLoader(dataloader, batch_size=BATCH_SIZE, shuffle=True,
                                                   num_workers=DATALOADER_WORKER)

    size = len(dataloader.dataset)
    rolling_average = RollingAverage(10)
    t = tqdm(enumerate(dataloader), desc='train_loop', total=size)
    for batch, (X, Y, _) in t:
        # Compute prediction and loss
        pred = model(X)
        # print(f"pred: {pred.shape}, Y:{Y[0].shape}")

        loss = loss_fn(pred, Y[0])
        # print(loss)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        t.set_description(f"loss: {rolling_average(loss.detach())}")


def validation(dataloader, model, loss_fn):
    """
    validation
    :param dataloader: Dataloader with data to validate on
    :param model: model to validate
    :param loss_fn: loss_fn to validate with
    :return: None
    """
    # TODO: can I add a accuracy function?

    dataloader = torch.utils.data.DataLoader(dataloader, batch_size=BATCH_SIZE, shuffle=True,
                                                  num_workers=DATALOADER_WORKER)
    size = len(dataloader.dataset)
    loss_sum = 0
    for batch, (X, Y, _) in tqdm(enumerate(dataloader), desc='validation_loop', total=size):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, Y[0]).detach().numpy()
        loss_sum += loss

    print(f"average loss: {loss_sum / size}")


if __name__ == '__main__':
    train('data/Training/', 'data/Validation/', load_model='models/model.pt', save_model='models/model.pt')