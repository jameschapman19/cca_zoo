"""
Gen-Oja: A Simple and Efficient Algorithm for
Streaming Generalized Eigenvector Computation
https://proceedings.neurips.cc/paper/2018/file/1b318124e37af6d74a03501474f44ea1-Paper.pdf
"""
from functools import partial
from os import environ

import jax
import jax.numpy as jnp
import optax
from ._utils import _get_AB, _gram_schmidt, _split_eigenvector
from jax import jit
from .._baseexperiment import _BaseExperiment
from ._ccamixin import _CCAMixin


class GenOja(_BaseExperiment,_CCAMixin):
    def __init__(self, mode, init_rng, config):
        super(GenOja, self).__init__(mode, init_rng, config)
        """Constructs the experiment.
        Args:
          mode: A string, equivalent to FLAGS.jaxline_mode when running normally.
          init_rng: A `PRNGKey` to use for experiment initialization.
        """
        """Initialization function for a Jaxline experiment."""
        self._W = jax.random.normal(
            self.init_rng, (self.n_components, self.dims[0] + self.dims[1])
        )
        self._Z = jax.random.normal(
            self.init_rng, (self.n_components, self.dims[0] + self.dims[1])
        )
        self._Z = self._Z / jnp.linalg.norm(self._Z, keepdims=True, axis=1)

        self._grads = jax.jit(
            jax.vmap(
                self._grads,
                in_axes=(None, None, 0, 0),
            )
        )
        self._update_with_grads_ls = jax.jit(
            jax.vmap(
                self._update_with_grads_ls,
                in_axes=(0, 0, 0),
            )
        )
        self._update_with_grads_oja = jax.jit(
            jax.vmap(
                self._update_with_grads_oja,
                in_axes=(0, 0, 0),
            )
        )
        self._optimizer_ls = optax.sgd(learning_rate=alpha)
        self._optimizer_oja = optax.sgd(learning_rate=beta0)
        self._opt_state_ls = self._optimizer_ls.init(self.W)
        self._opt_state_oja = self._optimizer_oja.init(self.V)

    def _update(self, views, global_step):
        X_i, Y_i = views
        w_grad = self._grads(X_i, Y_i, self._W, self._Z)
        self._W, self._opt_state_ls = self._update_with_grads_ls(
            self._W, w_grad, self._opt_state_ls
        )
        self._Z, self._opt_state_oja = self._update_with_grads_oja(
            self._Z, self._W / (global_step + 1), self._opt_state_oja
        )
        self._Z = _gram_schmidt(self._Z, _get_AB(X_i, Y_i)[1])
        self._U, self._V = _split_eigenvector(self._Z, self.dims[0])

    @staticmethod
    def _grads(X_i, Y_i, W, V):
        A, B = _get_AB(X_i, Y_i)
        return B @ W - A @ V

    @partial(jit, static_argnums=(0))
    def _update_with_grads_ls(self, wi, grads, opt_state):
        # we have gradient of utilities so we negate for gradient descent
        updates, opt_state = self._optimizer_ls.update(grads, opt_state)
        wi_new = optax.apply_updates(wi, updates)
        return wi_new, opt_state

    @partial(jit, static_argnums=(0))
    def _update_with_grads_oja(self, vi, grads, opt_state):
        # we have gradient of utilities so we negate for gradient descent
        updates, opt_state = self._optimizer_oja.update(-grads, opt_state)
        vi_new = optax.apply_updates(vi, updates)
        return vi_new, opt_state
