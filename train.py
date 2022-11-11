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


def train(load_model=None, save_model=None):
    """
    trainingsfunction
    :param load_model: (default: None) path to model to load
    :param save_model: (default: None) path to save the model
    :return: None
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create model
    model = DhSegment([3, 4, 6, 4], in_channels=IN_CHANNELS, out_channel=OUT_CHANNELS, load_resnet_weights=True)
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
    train_set, validation_set, _ = dataset.random_split([.9, .05, .05])
    print(f"train size: {len(train_set)}, test size: {len(validation_set)}")

    print(f"ration between classes: {train_set.class_ratio(OUT_CHANNELS)}")

    # set optimizer and loss_fn
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    loss_fn = CrossEntropyLoss()  # weight=torch.tensor(LOSS_WEIGHTS)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                                               num_workers=DATALOADER_WORKER)
    val_loader = DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=DATALOADER_WORKER)

    train_loop(train_loader, len(train_set), model, loss_fn, optimizer, validation_set, val_loader)

    model.save(save_model)

    # validation
    validation(validation_set, model, loss_fn, val_loader)


def train_loop(train_loader: DataLoader, n_train: int, model: torch.nn.Module, loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer, validation_set: NewsDataset, val_loader: DataLoader):
    """
    executes one training epoch
    :param train_loader: Dataloader object
    :param n_train: size of train data
    :param model: model to train
    :param loss_fn: loss function to optimize
    :param optimizer: optimizer to use
    :param validation_set: data for validation
    :return: None
    """

    for epoch in range(1, EPOCHS + 1):
        model.train()

        with tqdm.tqdm(total=n_train, desc=f'Epoch {epoch}/{EPOCHS}', unit='img') as pbar:
            for images, true_masks, _ in train_loader:

                images = images.to(DEVICE)
                true_masks = true_masks.to(DEVICE)

                # Compute prediction and loss
                pred = model(images)
                loss = loss_fn(pred, true_masks)
                # Backpropagation
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                EXPERIMENT.log({'train_images': wandb.Image(np.squeeze(images[0]).cpu()),
                                'train_masks': {
                                    'true': wandb.Image(np.squeeze(true_masks[0]).float().cpu()),
                                    'pred': wandb.Image(pred.argmax(dim=1)[0].float().cpu())
                                },
                                'train_loss': loss.item()})

                # update description
                pbar.update(images.shape[0])
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                del images, true_masks, pred, loss
                torch.cuda.empty_cache()

        validation(validation_set, model, loss_fn, val_loader)


def validation(data: NewsDataset, model, loss_fn, val_loader: DataLoader):
    """
    validation
    :param data: Dataloader with data to validate on
    :param model: model to validate
    :param loss_fn: loss_fn to validate with
    :return: None
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()

    size = len(val_loader)

    loss_sum = 0
    jaccard_sum = 0
    accuracy_sum = 0
    for images, true_masks, _ in tqdm.tqdm(val_loader, desc='validation_loop', total=size):
        # Compute prediction and loss
        images = images.to(device)
        true_masks = true_masks.to(device)

        pred = model(images)
        loss = loss_fn(pred, true_masks)

        pred = pred.detach().cpu().numpy()
        loss = loss.detach().cpu().numpy()
        # images = images.detach().cpu().numpy()
        true_masks = true_masks.detach().cpu().numpy()

        loss_sum += loss
        pred = np.argmax(pred, axis=1)
        jaccard_sum += sklearn.metrics.jaccard_score(true_masks.flatten(), pred.flatten(), average='macro')
        accuracy_sum += sklearn.metrics.accuracy_score(true_masks.flatten(), pred.flatten())

        del images, true_masks, pred, loss
        torch.cuda.empty_cache()

    images, true_masks, _ = val_loader.dataset[0]

    EXPERIMENT.log({'validation_images': wandb.Image(np.squeeze(images).cpu()),
                    'validation_masks': {
                        'true': wandb.Image(np.squeeze(true_masks).float().cpu()),
                    },
                    'validation_loss': loss_sum / size,
                    'validation_accuracy': accuracy_sum / size,
                    'validation_score': jaccard_sum / size, })
    print(f"average loss: {loss_sum / size}")
    print(f"average accuracy: {accuracy_sum / size}")
    print(f"average jaccard score: {jaccard_sum / size}")  # Intersection over Union


if __name__ == '__main__':
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # initialize wandb
    EXPERIMENT = wandb.init(project='newspaper-segmentation', entity="newspaper-segmentation", resume='allow',
                            anonymous='must')
    train(load_model=None, save_model='Models/model.pt')
