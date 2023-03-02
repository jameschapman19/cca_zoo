from abc import abstractmethod
from typing import Iterable, Union

import numpy as np
from scipy.linalg import block_diag
from torch.utils import data
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler

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
        learning_rate=0.1,
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
        self.dtypes = [np.float32]

    def fit(self, views: Iterable[np.ndarray], y=None, **kwargs):
        views = self._validate_inputs(views)
        self._check_params()
        train_dataloader, val_dataloader = self.get_dataloader(views)
        self.track = []
        initializer = _default_initializer(
            views, self.initialization, self.random_state, self.latent_dims
        )
        self.weights = initializer.fit(views).weights
        self.weights = [weights.astype(np.float32) for weights in self.weights]
        self.u= [w.copy() for w in self.weights]
        if self.nesterov:
            self.weights_old = self.weights.copy()
            self.lam = [0, 0]
        stop = False
        for _ in range(self.epochs):
            for i, sample in enumerate(train_dataloader):
                stop = self._update(sample["views"])
                if self.nesterov:
                    self._step_lambda()
            if self.val_split is not None:
                for i, sample in enumerate(val_dataloader):
                    self.track.append(
                        self.objective(sample["views"], weights=self.weights)
                    )
            else:
                self.track.append(self.objective(views, weights=self.weights))
            if stop:
                break
        return self

    def get_dataloader(self, views: Iterable[np.ndarray]):
        if self.batch_size is None:
            dataset = BatchNumpyDataset(views)
        else:
            dataset = NumpyDataset(views)
        if self.val_split is not None:
            train_size = int((1 - self.val_split) * len(dataset))
            val_size = len(dataset) - train_size
            dataset, val_dataset = data.random_split(dataset, [train_size, val_size])
            sampler = BatchSampler(
                SequentialSampler(dataset), batch_size=len(val_dataset), drop_last=False
            )
            val_loader = DataLoader(
                val_dataset,
                sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=self.drop_last,
                timeout=self.timeout,
                worker_init_fn=self.worker_init_fn,
                collate_fn=lambda x: x[0],
            )
        else:
            val_loader = None
        if self.batch_size is None:
            batch_size = len(dataset)
        else:
            batch_size = self.batch_size
        sampler = BatchSampler(
            RandomSampler(dataset), batch_size=batch_size, drop_last=False
        )
        train_loader = DataLoader(
            dataset,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            timeout=self.timeout,
            worker_init_fn=self.worker_init_fn,
            collate_fn=lambda x: x[0],
        )
        return train_loader, val_loader

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
    def objective(self, views, weights):
        raise NotImplementedError


class BatchNumpyDataset(NumpyDataset):
    def __getitem__(self, index):
        if self.labels is not None:
            return {"views": self.views, "label": self.labels}
        else:
            return {"views": self.views}


def tv(z):
    all_z = np.hstack(z)
    C = np.cov(all_z, rowvar=False)
    C -= block_diag(*[np.cov(z_, rowvar=False) for z_ in z])
    C /= z[0].shape[0]
    return np.linalg.svd(C, compute_uv=False).sum()


def tcc(z):
    return CCA(z[0].shape[1]).fit(z).score(z).sum()
