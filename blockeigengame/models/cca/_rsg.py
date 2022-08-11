import jax
import jax.numpy as jnp
from jax import jit
from pymanopt.manifolds import Grassmann

from ._ccamixin import _CCAMixin
from .._baseexperiment import _BaseExperiment


class RSG(_CCAMixin, _BaseExperiment):
    def __init__(self, mode, init_rng, config):
        super(RSG, self).__init__(mode, init_rng, config)

    def _init_train(self):
        self._init_ground_truth()
        views = next(self._train_input)
        self._U = jax.random.normal(
            self.init_rng, (self.config.n_components, views[0].shape[1])
        )
        self._U /= jnp.linalg.norm(self._U, axis=1, keepdims=True)
        self._V = jax.random.normal(
            self.init_rng, (self.config.n_components, views[1].shape[1])
        )
        self._V /= jnp.linalg.norm(self._V, axis=1, keepdims=True)
        self.Qx = jnp.linalg.qr(
            jax.random.normal(
                self.init_rng, (self.config.n_components, views[0].shape[1])
            )
        )
        self.Qy = jnp.linalg.qr(
            jax.random.normal(
                self.init_rng, (self.config.n_components, views[1].shape[1])
            )
        )
        self.aux_input = self._train_input.copy()
        self.Qx, self.Qy = self.streaming_pca()
        self.Tx = jnp.eye(self.config.n_components)
        self.Ty = jnp.eye(self.config.n_components)
        self.manx = Grassmann(views[0].shape[1], self.config.n_components)
        self.many = Grassmann(views[1].shape[1], self.config.n_components)

    def _update(self, views, global_step):
        (
            X_i,
            Y_i,
        ) = views
        Vx = X_i.T @ (-Y_i @ self.Qy @ Ty) @ Tx.T
        Vy = Y_i.T @ (-X_i @ self.Qx @ Tx) @ Ty.T
        tx = jnp.zeros((X_i.shape[1], self.config.n_components))
        ty = jnp.zeros((X_i.shape[1], self.config.n_components))
        Xn = jnp.reshape(
            (
                X_i.shape[1],
                self.config.batch_size,
                self.config.batch_size / self.config.n_components,
            )
        )
        Yn = jnp.reshape(
            (
                Y_i.shape[1],
                self.config.batch_size,
                self.config.batch_size / self.config.n_components,
            )
        )
        for j in range(Xn.shape[1]):
            q = Xn[:, :, j]
            tx = tx - self.manx.re
            q = Yn[:, :, j]
            ty = ty  # TODO
        Vx += white @ tx
        Vy += white @ ty
        Vx -= Qx @ Vx.T @ Qx
        Vy -= Qy @ Vy.T @ Qy
        Vx /= jnp.linalg.norm(Vx, keepdims=True, axis=0)
        Vy /= jnp.linalg.norm(Vy, keepdims=True, axis=0)
        Vtx = Qx.T @ X_i.T @ (-Y @ self.Qx @ Ty)
        Vty = -Qy.T @ Y_i.T @ (-X @ self.Qy @ TX)
        self.Qx = self.manx.retraction(self.Qx, -self.config.learning_rate * Vx)
        self.Qy = self.many.retraction(self.Qy, -self.config.learning_rate * Vy)
        Vtx -= Vtx.T
        Vty -= Vty.T
        Vtx /= jnp.linalg.norm(Vtx, keepdims=True, axis=0) * 2
        Vty /= jnp.linalg.norm(Vty, keepdims=True, axis=0) * 2
        Tx = Tx @ self.manx.retraction(
            jnp.eye(self.config.n_components), -self.config.learning_rate * Vtx
        )
        Ty = Ty @ self.many.retraction(
            jnp.eye(self.config.n_components), -self.config.learning_rate * Vty
        )

    def streaming_pca(self):
        for Xi, Yi in self.aux_input:
            self.Qx = self._pca_update(
                jnp.sqrt(self.config.learning_rate) * Xi, self.Qx
            )
            self.Qy = self._pca_update(
                jnp.sqrt(self.config.learning_rate) * Yi, self.Qy
            )

    @jit
    @staticmethod
    def _pca_update(X, Q):
        Q += X.T @ X @ Q
        return jnp.linalg.qr(Q)[0]
