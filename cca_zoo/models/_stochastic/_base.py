from abc import abstractmethod
from typing import Iterable

import numpy as np
from torch.utils import data

from cca_zoo.data import CCA_Dataset
from cca_zoo.models._base import _BaseCCA
from cca_zoo.utils import _check_views


class _BaseStochastic(_BaseCCA):
    r"""
    A class used to fit Stochastic PLS

    :Maths:

    .. math::


    :Citation:



    :Example:

    """

    def __init__(
        self,
        latent_dims: int = 1,
        scale: bool = True,
        centre=True,
        copy_data=True,
        random_state=None,
        eps=1e-3,
        accept_sparse=None,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        epochs=1,
        val_split=None,
        val_interval=10,
    ):
        """
        Constructor for rCCA

        :param latent_dims: number of latent dimensions to fit
        :param scale: normalize variance in each column before fitting
        :param centre: demean data by column before fitting (and before transforming out of sample
        :param copy_data: If True, views will be copied; else, it may be overwritten
        :param random_state: Pass for reproducible output across multiple function calls
        :param eps: epsilon for stability
        :param accept_sparse: which forms are accepted for sparse data
        """
        if accept_sparse is None:
            accept_sparse = ["csc", "csr"]
        super().__init__(
            latent_dims=latent_dims,
            scale=scale,
            centre=centre,
            copy_data=copy_data,
            accept_sparse=accept_sparse,
            random_state=random_state,
        )
        self.eps = eps
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.epochs = epochs
        self.val_interval = val_interval
        self.val_split = val_split

    def fit(self, views: Iterable[np.ndarray], y=None, **kwargs):
        """
        Fits a regularised CCA (canonical ridge) model

        :param views: list/tuple of numpy arrays or array likes with the same number of rows (samples)
        """
        views = _check_views(
            *views, copy=self.copy_data, accept_sparse=self.accept_sparse
        )
        scaled_views = self._centre_scale(views)
        dataset = CCA_Dataset(scaled_views)
        dataloader = data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            sampler=self.sampler,
            batch_sampler=self.batch_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            timeout=self.timeout,
            worker_init_fn=self.worker_init_fn,
        )
        self.track = []
        for _ in range(self.epochs):
            for i, sample in enumerate(dataloader):
                self.update([view.numpy() for view in sample["views"]])
                if i % self.val_interval == 0:
                    self.track.append(self.objective(views))
        return self

    @abstractmethod
    def update(self, views):
        pass

    @abstractmethod
    def objective(self, views, **kwargs):
        return 0
