import time
from abc import abstractmethod

import jax.numpy as jnp
import wandb
from jax import random
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    MultiOutputMixin,
    RegressorMixin,
)
from sklearn.model_selection import train_test_split


class _PCA(BaseEstimator, TransformerMixin, MultiOutputMixin, RegressorMixin):
    def __init__(self, n_components=2, *, scale=True, copy=True, wandb=True, verbose=False, random_state=None):
        self.n_components = n_components
        self.scale = scale
        self.copy = copy
        self.wandb = wandb
        self.verbose = verbose
        self.random_state = random_state
        self.scikit_random_state = random_state

    @abstractmethod
    def _fit(self, X, X_val):
        raise NotImplementedError

    @abstractmethod
    def fit(self, X):
        X, X_val = train_test_split(
            X, random_state=self.random_state, train_size=0.9
        )
        X, self._x_mean, self._x_std, = self.center_scale(
            X
        )
        start_time = time.time()
        self.x_weights = self._fit(X, X_val)
        self.fit_time = time.time() - start_time
        return self

    def score(self, X, y=None, sample_weight=None):
        X = self.transform(X)
        return self.TV(X)

    @abstractmethod
    def transform(self, X):
        raise NotImplementedError

    @staticmethod
    def initialize(X, n, type="uniform", random_state=None):
        if type == "uniform":
            V1 = jnp.ones((X.shape[1], n))
            V1 = V1 / jnp.linalg.norm(V1, axis=0)
        elif type == "random":
            V1 = random.normal(random_state, (X.shape[1], n))
            V1 = V1 / jnp.linalg.norm(V1, axis=0)
        else:
            print(f'Initialization "{type}" not implemented')
            return
        return V1

    def callback(self, obj_tr, obj_val, iteration, start_time=None):
        if self.wandb:
            wandb.log({"Iteration/Objective (Train)": obj_tr,
                       "Iteration/Objective (Val)": obj_val}, step=iteration)
        else:
            self.obj.append([obj_tr, obj_val])
        if self.verbose:
            if start_time is not None:
                epoch_time = time.time() - start_time
                print(f"Epoch {iteration} in {epoch_time} sec")
            print(f"Epoch {iteration} objective (Train): {obj_tr}")
            print(f"Epoch {iteration} objective (Train): {obj_val}")

    @staticmethod
    def TV(X):
        eigvals = jnp.linalg.eigvalsh(X.T @ X)
        return eigvals.real.sum()

    def center_scale(self, X):
        x_mean = X.mean(axis=0)
        X -= x_mean
        x_std = X.std(axis=0)
        if self.scale:
            X /= x_std
        return X, x_mean, x_std
