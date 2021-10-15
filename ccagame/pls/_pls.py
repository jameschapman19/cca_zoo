import time
from abc import abstractmethod

import jax.numpy as jnp
import jax.scipy as jsp
from jax import random
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    MultiOutputMixin,
    RegressorMixin,
)

from ..utils import check_random_state


class _PLS(BaseEstimator, TransformerMixin, MultiOutputMixin, RegressorMixin):
    def __init__(self, n_components=2, *, scale=True, copy=True, wandb=True, verbose=False, random_state=None):
        self.n_components = n_components
        self.scale = scale
        self.copy = copy
        self.wandb = wandb
        self.verbose = verbose
        self.random_state = check_random_state(random_state)

    @abstractmethod
    def _fit(self, X, Y):
        raise NotImplementedError

    @abstractmethod
    def fit(self, X, Y):
        X, Y, self._x_mean, self._y_mean, self._x_std, self._y_std = self.center_scale(
            X, Y
        )
        start_time = time.time()
        self.x_weights, self.y_weights = self._fit(X, Y)
        self.fit_time = time.time() - start_time
        return self

    @abstractmethod
    def transform(self, X, Y):
        raise NotImplementedError

    def score(self, X, y, sample_weight=None):
        return self.TV(X, y, self.x_weights, self.y_weights)

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

    @staticmethod
    def TV(X, Y, Wx, Wy):
        X_hat = X @ Wx
        Y_hat = Y @ Wy
        C = X_hat.T @ Y_hat
        _, S, _ = jnp.linalg.svd(C)
        return S.sum()

    @staticmethod
    def TCC(X, Y, Wx, Wy):
        dof = X.shape[0] - 1
        X_hat = X @ Wx
        Y_hat = Y @ Wy
        C = jnp.hstack((X_hat, Y_hat))
        C = C.T @ C / dof
        # Get the block covariance matrix placing Xi^TX_i on the diagonal
        D = jsp.linalg.block_diag(*[m.T @ m for i, m in enumerate([X_hat, Y_hat])]) / dof
        C = C - jsp.linalg.block_diag(*[view.T @ view / dof for view in [X_hat, Y_hat]]) + D
        R = jnp.linalg.inv(jnp.linalg.cholesky(D))
        # In MCCA our eigenvalue problem Cv = lambda Dv
        C_whitened = R @ C @ R.T
        eigvals = jnp.linalg.eigvalsh(C_whitened)[::-1][: Wx.shape[1]] - 1
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
