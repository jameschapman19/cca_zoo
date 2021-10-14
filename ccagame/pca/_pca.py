import time
from abc import abstractmethod

import jax.numpy as jnp
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    MultiOutputMixin,
    RegressorMixin,
)

from .utils import TV


class _PCA(BaseEstimator, TransformerMixin, MultiOutputMixin, RegressorMixin):
    def __init__(self, n_components=2, *, scale=True, copy=True, wandb=True, verbose=False, random_state=None):
        self.n_components = n_components
        self.scale = scale
        self.copy = copy
        self.wandb = wandb
        self.verbose = verbose
        self.random_state = random_state

    @abstractmethod
    def _fit(self, X):
        raise NotImplementedError

    @abstractmethod
    def fit(self, X):
        self.mean = jnp.mean(X, axis=0)
        X -= self.mean
        start_time = time.time()
        self.x_weights = self._fit(X)
        self.fit_time = time.time() - start_time
        return self

    def score(self, X, y=None, sample_weight=None):
        return TV(X, self.x_weights)

    @abstractmethod
    def transform(self, X):
        raise NotImplementedError
