import time
from abc import abstractmethod

import jax.numpy as jnp
import jax.scipy as jsp
import sklearn.utils
import wandb
from jax import random
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    MultiOutputMixin,
    RegressorMixin,
)
from sklearn.model_selection import train_test_split

from ..utils import check_random_state


class _PLS(BaseEstimator, TransformerMixin, MultiOutputMixin, RegressorMixin):
    def __init__(self, n_components=2, *, scale=True, copy=True, wandb=True, verbose=False, random_state=None):
        self.n_components = n_components
        self.scale = scale
        self.copy = copy
        self.wandb = wandb
        self.verbose = verbose
        self.random_state = check_random_state(random_state)
        self.scikit_random_state = sklearn.utils.check_random_state(random_state)
        self.obj = []

    @abstractmethod
    def _fit(self, X, Y, X_val=None, Y_val=None):
        raise NotImplementedError

    @abstractmethod
    def fit(self, X, Y):
        X, X_val, Y, Y_val = train_test_split(
            X, Y, random_state=self.scikit_random_state, train_size=0.9
        )
        X, Y, self._x_mean, self._y_mean, self._x_std, self._y_std = self.center_scale(
            X, Y
        )
        X_val = (X_val - self._x_mean) / self._x_std
        Y_val = (Y_val - self._y_mean) / self._y_std
        start_time = time.time()
        self.x_weights, self.y_weights = self._fit(X, Y, X_val, Y_val)
        self.fit_time = time.time() - start_time
        return self

    @abstractmethod
    def transform(self, X, Y):
        X = (X - self._x_mean) / self._x_std
        Y = (Y - self._y_mean) / self._y_std
        return X @ self.x_weights, Y @ self.y_weights

    def score(self, X, y, sample_weight=None):
        X, Y = self.transform(X, y)
        return self.TV(X, Y)

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

    @staticmethod
    def initialize(X, Y, k, type="uniform", random_state=None):
        if type == "svd":
            U1, _, V1 = jnp.linalg.svd(X.T @ Y)
            U1 = U1[:, :k]
            V1 = V1[:, :k]
        elif type == "uniform":
            U1 = jnp.ones((X.shape[1], k))
            V1 = jnp.ones((Y.shape[1], k))
            U1 = U1 / jnp.linalg.norm(U1, axis=0)
            V1 = V1 / jnp.linalg.norm(V1, axis=0)
        elif type == "random":
            key, subkey = random.split(random_state)
            U1 = random.normal(key, (X.shape[1], k))
            V1 = random.normal(subkey, (Y.shape[1], k))
            U1 = U1 / jnp.linalg.norm(U1, axis=0)
            V1 = V1 / jnp.linalg.norm(V1, axis=0)
        else:
            print(f'Initialization "{type}" not implemented')
            return
        return U1, V1

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
    def TV(X, Y):
        C = X.T @ Y
        _, S, _ = jnp.linalg.svd(C)
        return S.sum()

    @staticmethod
    def TCC(X, Y):
        dof = X.shape[0] - 1
        C = jnp.hstack((X, Y))
        C = C.T @ C / dof
        # Get the block covariance matrix placing Xi^TX_i on the diagonal
        D = jsp.linalg.block_diag(*[m.T @ m for i, m in enumerate([X, Y])]) / dof
        C = C - jsp.linalg.block_diag(*[view.T @ view / dof for view in [X, Y]]) + D
        R = jnp.linalg.inv(jnp.linalg.cholesky(D))
        # In MCCA our eigenvalue problem Cv = lambda Dv
        C_whitened = R @ C @ R.T
        eigvals = jnp.linalg.eigvalsh(C_whitened)[::-1][: X.shape[1]] - 1
        return eigvals.real.sum()

    @staticmethod
    def gram_schmidt_matrix(W, M):
        for k in range(W.shape[1]):
            C = jnp.zeros((W.shape[0], k))
            for j in range(k):
                C = C.at[:, j].set(
                    jnp.dot(W[:, j], jnp.dot(jnp.transpose(W[:, k]), jnp.dot(M, W[:, j])))
                )
            W = W.at[:, k].set(W[:, k] - jnp.sum(C, axis=1))
            W = W.at[:, k].set(
                W[:, k] / jnp.sqrt(jnp.dot(jnp.transpose(W[:, k]), jnp.dot(M, W[:, k])))
            )
        return W

    @staticmethod
    def initialize_gep(X, Y):
        n = X.shape[0]
        A = jnp.hstack((X, Y))
        A = jnp.dot(jnp.transpose(A), A) / n
        B = (
                jsp.linalg.block_diag(
                    jnp.dot(jnp.transpose(X), X), jnp.dot(jnp.transpose(Y), Y)
                )
                / n
        )
        A = A - B
        return A, B
