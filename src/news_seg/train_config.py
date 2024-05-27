"""Default training config containing training parameters and model input and output sizes."""
import torch

EPOCHS = 1
DATALOADER_WORKER = 1
IN_CHANNELS, OUT_CHANNELS = 3, 10
VAL_NUMBER = 3
DEFAULT_SPLIT = (0.85, 0.05, 0.10)

BATCH_SIZE = 32
LEARNING_RATE = 1e-5  # 1e-5 from Paper .001 Standard 0,0001 seems to work well
WEIGHT_DECAY = 1e-6  # 1e-6 from Paper
LOSS_WEIGHTS: torch.Tensor = torch.tensor([
    5.0,  # background
    10.0,  # caption
    10.0,  # table
    2.0,  # paragraph
    20.0,  # heading
    10.0,  # header
    20.0,  # separator_vertical
    20.0,  # separator
    10.0,  # image
    10.0  # inverted_text
]) / 20
