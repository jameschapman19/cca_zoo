from typing import Iterable

import numpy as np
from torch.utils.data import Dataset, DataLoader


class NumpyDataset(Dataset):
    """
    Class that turns numpy arrays into a torch dataset
    """

    def __init__(self, views, labels=None, scale=False, centre=False):
        """

        :param views: list/tuple of numpy arrays or array likes with the same number of rows (samples)
        """
        self.labels = labels
        self.centre = centre
        self.scale = scale
        self.views = self._centre_scale(views)

    def __len__(self):
        return len(self.views[0])

    def __getitem__(self, index):
        views = [view[index].astype(np.float32) for view in self.views]
        if self.labels is not None:
            label = self.labels[index]
            return {"views": views, "label": label}
        else:
            return {"views": views}

    def _centre_scale(self, views: Iterable[np.ndarray]):
        """
        Centers and scales the data

        Parameters
        ----------
        views : list/tuple of numpy arrays or array likes with the same number of rows (samples)

        Returns
        -------
        views : list of numpy arrays


        """
        self.view_means = []
        self.view_stds = []
        transformed_views = []
        for view in views:
            if self.centre:
                view_mean = view.mean(axis=0)
                self.view_means.append(view_mean)
                view = view - self.view_means[-1]
            if self.scale:
                view_std = view.std(axis=0, ddof=1)
                view_std[view_std == 0.0] = 1.0
                self.view_stds.append(view_std)
                view = view / self.view_stds[-1]
            transformed_views.append(view)
        return transformed_views


def check_dataset(dataset):
    """
    Checks that a custom dataset returns a dictionary with a "views" key containing a list of tensors

    Parameters
    ----------
    dataset: torch.utils.data.Dataset

    Returns
    -------

    """
    dataloader = DataLoader(
        dataset,
    )
    for batch in dataloader:
        if "views" not in batch:
            raise ValueError(
                "The dataset must return a dictionary with a 'views' key containing a list of tensors"
            )
        else:
            break


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
    persistent_workers=True,
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
    :param persistent_workers: whether to keep workers alive after dataloader is destroyed

    """
    if num_workers == 0:
        persistent_workers = False
    if batch_size is None:
        batch_size = len(dataset)
    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle_train,
        persistent_workers=persistent_workers,
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
            persistent_workers=persistent_workers,
        )
        return train_dataloader, val_dataloader
    return train_dataloader
