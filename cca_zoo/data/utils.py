import numpy as np
from torch.utils.data import Dataset


class CCA_Dataset(Dataset):
    """
    Class that turns numpy arrays into a torch dataset

    """

    def __init__(self, *views, labels=None, train=True):
        """

        :param views:
        :param labels:
        :param train:
        """
        self.train = train
        self.views = [view for view in views]
        if labels is None:
            self.labels = np.zeros(len(self.views[0]))
        else:
            self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        views = [view[idx] for view in self.views]
        return tuple(views), label
