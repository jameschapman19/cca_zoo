from abc import abstractmethod

from sklearn.base import BaseEstimator, TransformerMixin, MultiOutputMixin, RegressorMixin

from .utils import TV
import time

class _PLS(BaseEstimator, TransformerMixin, MultiOutputMixin, RegressorMixin):
    def __init__(self, n_components=2, *, scale=True, copy=True):
        self.n_components = n_components
        self.scale = scale
        self.copy = copy

    @abstractmethod
    def _fit(self, X, Y):
        raise NotImplementedError

    @abstractmethod
    def fit(self, X, Y):
        X, Y, self._x_mean, self._y_mean, self._x_std, self._y_std = self.center_scale(X, Y)
        start_time = time.time()
        self.x_weights, self.y_weights = self._fit(X, Y)
        self.fit_time=time.time()-start_time
        return self

    @abstractmethod
    def transform(self, X, Y):
        raise NotImplementedError

    def score(self, X, y, sample_weight=None):
        return TV(X, y, self.x_weights, self.y_weights)

    def center_scale(self, X, Y):
        x_mean = X.mean(axis=0)
        y_mean = Y.mean(axis=0)
        X -= x_mean
        Y -= y_mean
        x_std = X.std(axis=0)
        y_std = Y.std(axis=0)
        if self.scale:
            X /= x_std
            Y /= y_std
        return X, Y, x_mean, y_mean, x_std, y_std
