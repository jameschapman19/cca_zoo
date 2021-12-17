import numpy as np
from torch.utils.data import Dataset


class CCA_Dataset(Dataset):
    """
    Class that turns numpy arrays into a torch dataset

    """

    def __init__(self, views):
        """

        :param views: list/tuple of numpy arrays or array likes with the same number of rows (samples)
        """
        self.views = [view for view in views]

    def __len__(self):
        return len(self.views[0])

    def __getitem__(self, idx):
        views = [view[idx].astype(np.float32) for view in self.views]
        return {"views": views}
