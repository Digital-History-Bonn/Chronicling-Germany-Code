"""shared utility functions"""
import torch
from torch import randperm


def initialize_random_split(length, ratio):
    assert sum(ratio) == 1, "ratio does not sum up to 1."
    assert len(ratio) == 3, "ratio does not have length 3"
    assert (
            int(ratio[0] * length) > 0
            and int(ratio[1] * length) > 0
            and int(ratio[2] * length) > 0
    ), (
        "Dataset is to small for given split ratios for test and validation dataset. "
        "Test or validation dataset have size of zero."
    )
    splits = int(ratio[0] * length), int(ratio[0] * length) + int(
        ratio[1] * length
    )
    indices = randperm(
        length, generator=torch.Generator().manual_seed(42)
    ).tolist()
    return indices, splits
