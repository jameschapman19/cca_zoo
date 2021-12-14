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
    dataset,
    val_dataset=None,
    batch_size=None,
    val_batch_size=None,
    drop_last=True,
    val_drop_last=False,
    shuffle_train=False,
    pin_memory=True,
    num_workers=0,
):
    """
    A utility function to allow users to quickly get hold of the dataloaders required by pytorch lightning

    :param dataset: A CCA dataset used for training
    :param val_dataset: An optional CCA dataset used for validation
    :param batch_size: batch size of train loader
    :param val_batch_size: batch size of val loader
    :param num_workers: number of workers used
    :param pin_memory: pin memory used by pytorch - True tends to speed up training
    :param shuffle_train: whether to shuffle training data
    :param val_drop_last: whether to drop the last incomplete batch from the validation data
    :param drop_last: whether to drop the last incomplete batch from the train data

    """
    if batch_size is None:
        batch_size = len(dataset)
    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle_train,
    )
    if val_dataset:
        if val_batch_size is None:
            val_batch_size = len(val_dataset)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            drop_last=val_drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        return train_dataloader, val_dataloader
    return train_dataloader
