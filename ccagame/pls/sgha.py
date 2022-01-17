"""
Gen-Oja: A Simple and Efficient Algorithm for
Streaming Generalized Eigenvector Computation
https://proceedings.neurips.cc/paper/2018/file/1b318124e37af6d74a03501474f44ea1-Paper.pdf
"""
from functools import partial
from os import environ

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import optax
from jax import jit
from numpy import linalg
from scipy.linalg.misc import norm

from . import PLSExperiment


class SGHA(PLSExperiment):
    NON_BROADCAST_CHECKPOINT_ATTRS = {"_U": "U", "_V": "V"}

    def __init__(
        self,
        mode,
        init_rng=None,
        num_devices=1,
        n_components=1,
        dims=None,
        data=None,
        learning_rate=1e-6,
        momentum=0.9,
        nesterov=True,
        batch_size=0,
        **kwargs
    ):
        super(SGHA, self).__init__(
            mode,
            init_rng=init_rng,
            num_devices=num_devices,
            dims=dims,
            n_components=n_components,
            data=data,
            batch_size=batch_size,
            **kwargs
        )
        """Constructs the experiment.
        Args:
          mode: A string, equivalent to FLAGS.jaxline_mode when running normally.
          init_rng: A `PRNGKey` to use for experiment initialization.
        """
        """Initialization function for a Jaxline experiment."""
        self._W = jax.random.normal(
            self.init_rng, (self.n_components, self.dims[0] + self.dims[1])
        )
        self._W /= jnp.linalg.norm(self._W, axis=1, keepdims=True)
        self._update_with_grads = jax.jit(
            jax.vmap(
                self._update_with_grads,
                in_axes=(0, 1, 0),
            )
        )
        self._optimizer = optax.sgd(learning_rate=learning_rate)
        self._opt_state = self._optimizer.init(self._W)

    def _update(self, views, global_step):
        X_i, Y_i = views
        w_grad = self._grad(X_i, Y_i, self._W)
        self._W, self._opt_state = self._update_with_grads(
            self._W, w_grad, self._opt_state
        )
        norm = jnp.linalg.norm(self._W, axis=1, keepdims=True)
        norm = norm.at[norm < 1].set(1)
        self._W /= norm
        self._U, self._V = self._split_eigenvector(self._W)

    @staticmethod
    @jit
    def _grad(X_i, Y_i, W):
        n = X_i.shape[0]
        A = (
            jnp.hstack((X_i, Y_i)).T @ jnp.hstack((X_i, Y_i))
            - jsp.linalg.block_diag(X_i.T @ X_i, Y_i.T @ Y_i)
        ) / n
        Y = W @ A @ W.T
        return W.T @ jnp.triu(Y) - A @ W.T

    @partial(jit, static_argnums=(0))
    def _update_with_grads(self, wi, grads, opt_state):
        updates, opt_state = self._optimizer.update(grads, opt_state)
        wi_new = optax.apply_updates(wi, updates)
        return wi_new, opt_state

    @partial(jit, static_argnums=(0))
    def _split_eigenvector(self, V):
        return V[:, : self.dims[0]], V[:, self.dims[0] :]
