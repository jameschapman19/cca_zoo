from abc import abstractmethod
from typing import Iterable

import numpy as np
from scipy.linalg import block_diag
from torch.utils import data

from cca_zoo.data.deep import NumpyDataset
from cca_zoo.models import CCA
from cca_zoo.models._base import _BaseCCA


class _BaseStochastic(_BaseCCA):
    def __init__(
        self,
        latent_dims: int = 1,
        scale: bool = True,
        centre=True,
        copy_data=True,
        random_state=None,
        eps=1e-3,
        accept_sparse=None,
        batch_size=None,
        shuffle=True,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        timeout=0,
        worker_init_fn=None,
        epochs=1,
        val_split=None,
        val_interval=10,
        learning_rate=0.01,
    ):
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
        self.learning_rate = learning_rate

    def fit(self, views: Iterable[np.ndarray], y=None, **kwargs):
        views = self._validate_inputs(views)
        self._check_params()
        dataset = NumpyDataset(views)
        if self.val_split is not None:
            train_size = int((1 - self.val_split) * len(dataset))
            val_size = len(dataset) - train_size
            dataset, val_dataset = data.random_split(dataset, [train_size, val_size])
        if self.batch_size is None:
            self.batch_size = len(dataset)
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
        if self.val_split is not None:
            val_dataloader = data.DataLoader(
                val_dataset,
                batch_size=len(val_dataset),
            )
        self.track = []
        self.weights = [
            self.random_state.normal(0, 1, size=(view.shape[1], self.latent_dims))
            for view in views
        ]
        # normalize weights
        self.weights = [
            weight / np.linalg.norm(weight, axis=0) for weight in self.weights
        ]
        for _ in range(self.epochs):
            for i, sample in enumerate(dataloader):
                self.update([view.numpy() for view in sample["views"]])
            if self.val_split is not None:
                for i, sample in enumerate(val_dataloader):
                    self.track.append(self.objective(sample["views"]))
        return self

    @abstractmethod
    def update(self, views):
        pass

    @abstractmethod
    def objective(self, views, **kwargs):
        return self.tcc(views)

    def tv(self, views):
        # q from qr decomposition of weights
        q = [np.linalg.qr(weight)[0] for weight in self.weights]
        views = self._centre_scale_transform(views)
        transformed_views = []
        for i, (view) in enumerate(views):
            transformed_view = view @ q[i]
            transformed_views.append(transformed_view)
        return tv(transformed_views)

    def tcc(self, views):
        z = self.transform(views)
        return tcc(z)


def tv(z):
    all_z = np.hstack(z)
    C = np.cov(all_z, rowvar=False)
    C -= block_diag(*[np.cov(z_, rowvar=False) for z_ in z])
    C /= z[0].shape[0]
    return np.linalg.svd(C, compute_uv=False).sum()


def tcc(z):
    return CCA(z[0].shape[1]).fit(z).score(z).sum()
