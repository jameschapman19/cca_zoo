import numpy as np
from sklearn.utils import check_random_state
from torch.utils.data import DataLoader, Dataset


class NumpyDataset(Dataset):
    """
    Class that turns numpy arrays into a torch dataset
    """

    def __init__(self, views, batch_size=None, random_state=None):
        self.views = [view.astype(np.float32) for view in views]
        self.batch_size = batch_size
        self.random_state = check_random_state(random_state)

    def __len__(self):
        return len(self.views[0])

    def __getitem__(self, index):
        views = [view[index] for view in self.views]
        independent_index = (
            index
            if self.batch_size is None
            else self.random_state.randint(0, len(self))
        )
        independent_views = [view[independent_index] for view in self.views]
        return {"views": views, "independent_views": independent_views}


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
                "The dataset must return a dictionary with a 'representations' key containing a list of tensors"
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
