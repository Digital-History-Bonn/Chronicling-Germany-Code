import torchsummary
import sklearn.metrics
import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import numpy as np
import tqdm
import wandb
from NewsDataset import NewsDataset
from model import DhSegment

EPOCHS = 5
BATCH_SIZE = 1
DATALOADER_WORKER = 1
IN_CHANNELS, OUT_CHANNELS = 1, 10
LEARNING_RATE = 0.0001  # 0,0001 seems to work well
# LOSS_WEIGHTS = [1.0, 9.0]  # 1 and 5 seems to work well

# set random seed for reproducibility
torch.manual_seed(42)

# initialize wandb
EXPERIMENT = wandb.init(project='newspaper-segmentation', entity="newspaper-segmentation", resume='allow',
                        anonymous='must')


def train(load_model=None, save_model=None):
    """
    trainingsfunction
    :param load_model: (default: None) path to model to load
    :param save_model: (default: None) path to save the model
    :return: None
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create model
    model = DhSegment([3, 3, 4, 3], in_channels=IN_CHANNELS, out_channel=OUT_CHANNELS, load_resnet_weights=True)
    # [3, 4, 6, 4]
    model = model.float()
    torchsummary.summary(model, input_size=(3, 64, 64), batch_size=-1)

    # load model if argument is None it does nothing
    model.load(load_model)

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

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                                               num_workers=DATALOADER_WORKER)

    train_loop(train_loader, len(train_set), model, loss_fn, optimizer)

    model.save(save_model)

    # validation
    validation(test_set, model, loss_fn)


def train_loop(train_loader: DataLoader, n_train: int, model: torch.nn.Module, loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer):
    """
    executes one training epoch
    :param train_loader: Dataloader object
    :param n_train: size of train data
    :param model: model to train
    :param loss_fn: loss function to optimize
    :param optimizer: optimizer to use
    :return: None
    """

    for epoch in range(1, EPOCHS + 1):
        model.train()
        # rolling_average = RollingAverage(10)

        with tqdm.tqdm(total=n_train, desc=f'Epoch {epoch}/{EPOCHS}', unit='img') as pbar:
            print(torch.cuda.memory_summary(device=DEVICE, abbreviated=False))
            for batch in train_loader:
                images = batch[0]
                true_masks = batch[1]

                # print(f"{X.shape=}")
                # print(f"start: {torch.cuda.memory_reserved()=}")
                images = images.to(DEVICE)
                true_masks = true_masks.to(DEVICE)

                # Compute prediction and loss
                pred = model(images)
                loss = loss_fn(pred, true_masks)
                # Backpropagation
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                EXPERIMENT.log({'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(pred.argmax(dim=1)[0].float().cpu()),
                                    'train_loss': loss.item()
                                }, })
                # print(f"optimizer: {torch.cuda.memory_reserved()=}")
                # update description
                pbar.update(images.shape[0])
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                del images, true_masks, pred, loss
                torch.cuda.empty_cache()
                # print(f"end: {torch.cuda.memory_reserved()=}")
                # print()


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
    for images, masks, _ in tqdm.tqdm(data, desc='validation_loop', total=size):
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
        jaccard_sum += sklearn.metrics.jaccard_score(masks.flatten(), pred.flatten(), average='macro')
        accuracy_sum += sklearn.metrics.accuracy_score(masks.flatten(), pred.flatten())

        del images, masks, pred, loss
        torch.cuda.empty_cache()

    print(f"average loss: {loss_sum / size}")
    print(f"average accuracy: {accuracy_sum / size}")
    print(f"average jaccard score: {jaccard_sum / size}")  # Intersection over Union


if __name__ == '__main__':
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    train(load_model=None, save_model='Models/model.pt')
