from abc import abstractmethod
from typing import Iterable, Union

import numpy as np
from scipy.linalg import block_diag
from torch.utils import data

from cca_zoo.data.deep import NumpyDataset
from cca_zoo.models import CCA
from cca_zoo.models._base import _BaseCCA
from cca_zoo.models._iterative._base import _default_initializer


class _BaseStochastic(_BaseCCA):
    def __init__(
        self,
        latent_dims: int = 1,
        scale: bool = True,
        centre=True,
        copy_data=True,
        random_state=None,
        tol=1e-3,
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
        learning_rate=0.01,
        initialization: Union[str, callable] = "random",
        track_training=False,
        nesterov=True,
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
        self.tol = tol
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
        self.val_split = val_split
        self.learning_rate = learning_rate
        self.initialization = initialization
        self.track_training = track_training
        self.nesterov = nesterov

    def fit(self, views: Iterable[np.ndarray], y=None, **kwargs):
        views = self._validate_inputs(views)
        self._check_params()
        train_dataloader, val_dataloader = self.get_dataloader(views)
        self.track = []
        initializer = _default_initializer(
            views, self.initialization, self.random_state, self.latent_dims
        )
        self.weights = initializer.fit(views).weights
        if self.nesterov:
            self.weights_old = self.weights.copy()
            self.lam = [0, 0]
        stop = False
        for _ in range(self.epochs):
            for i, sample in enumerate(train_dataloader):
                stop = self._update([v.numpy() for v in sample["views"]])
                if self.nesterov:
                    self._step_lambda()
                if self.batch_size is None:
                    self.objective([v.numpy() for v in sample["views"]])
            if self.val_split is not None:
                for i, sample in enumerate(val_dataloader):
                    self.track.append(
                        self.objective([v.numpy() for v in sample["views"]])
                    )
            if stop:
                break
        return self

    def get_dataloader(self, views: Iterable[np.ndarray]):
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
        else:
            val_dataloader = None
        return dataloader, val_dataloader

    @property
    def momentum(self):
        return (self.lam[0] - 1) / max(1, self.lam[1])

    def _step_lambda(self):
        self.lam[1] = self.lam[0]
        self.lam[0] = (1 + np.sqrt(1 + 4 * self.lam[1] ** 2)) / 2

    @abstractmethod
    def _update(self, views):
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
