import numpy as np
from torch.utils.data import Dataset


class CCA_Dataset(Dataset):
    """
    Class that turns numpy arrays into a torch dataset

    """

    def __init__(self, views, labels=None):
        """

        :param views: list/tuple of numpy arrays or array likes with the same number of rows (samples)
        :param labels: optional labels
        """
        self.views = [view for view in views]
        if labels is None:
            self.labels = np.zeros(len(self.views[0]))
        else:
            self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        views = [view[idx].astype(np.float32) for view in self.views]
        return tuple(views), label
