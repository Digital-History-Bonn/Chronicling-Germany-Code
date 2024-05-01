"""Default training config containing training parameters and model input and output sizes."""
import torch

EPOCHS = 1
DATALOADER_WORKER = 1
IN_CHANNELS, OUT_CHANNELS = 3, 10
VAL_NUMBER = 3
DEFAULT_SPLIT = (0.9, 0.05, 0.05)

BATCH_SIZE = 32
LEARNING_RATE = 1e-5  # 1e-5 from Paper .001 Standard 0,0001 seems to work well
WEIGHT_DECAY = 1e-6  # 1e-6 from Paper
LOSS_WEIGHTS: torch.Tensor = torch.tensor([
    2.0,
    10.0,
    10.0,
    10.0,
    4.0,
    20.0,
    10.0,
    10.0,
    10.0,
    20.0,
]) / 20
