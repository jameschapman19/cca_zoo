from abc import abstractmethod

import jax.numpy as jnp
from sklearn.base import BaseEstimator, TransformerMixin, MultiOutputMixin, RegressorMixin

from .utils import TV


class _PCA(BaseEstimator, TransformerMixin, MultiOutputMixin, RegressorMixin):
    def __init__(self, n_components=2, *, scale=True, copy=True):
        self.n_components = n_components
        self.scale = scale
        self.copy = copy

    @abstractmethod
    def _fit(self, X):
        raise NotImplementedError

    @abstractmethod
    def fit(self, X):
        self.mean = jnp.mean(X, axis=0)
        X -= self.mean
        self.x_weights = self._fit(X)
        return self

    def score(self, X, y, sample_weight=None):
        return TV(X, self.x_weights)

    @abstractmethod
    def transform(self, X):
        raise NotImplementedError
