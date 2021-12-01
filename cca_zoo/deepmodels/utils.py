from typing import Union, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from cca_zoo.data.utils import CCA_Dataset


def process_data(
        dataset: Union[torch.utils.data.Dataset, Iterable[np.ndarray]],
        val_dataset: Union[torch.utils.data.Dataset, Iterable[np.ndarray]] = None,
        labels=None,
        val_labels=None,
        val_split: float = 0,
):
    # Ensure datasets are in the right form (e.g. if numpy arrays are passed turn them into
    if isinstance(dataset, tuple):
        dataset = CCA_Dataset(dataset, labels=labels)
    if val_dataset is None and val_split > 0:
        lengths = [
            len(dataset) - int(len(dataset) * val_split),
            int(len(dataset) * val_split),
        ]
        dataset, val_dataset = torch.utils.data.random_split(dataset, lengths)
    elif isinstance(val_dataset, tuple):
        val_dataset = CCA_Dataset(val_dataset, labels=val_labels)
    return dataset, val_dataset


def get_dataloaders(
        dataset, val_dataset=None, batch_size=None, val_batch_size=None, num_workers=0
):
    if batch_size is None:
        batch_size = len(dataset)
    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )
    if val_dataset:
        if val_batch_size is None:
            val_batch_size = len(val_dataset)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return train_dataloader, val_dataloader
    return train_dataloader
