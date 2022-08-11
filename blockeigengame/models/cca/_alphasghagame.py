"""
Gen-Oja: A Simple and Efficient Algorithm for
Streaming Generalized Eigenvector Computation
https://proceedings.neurips.cc/paper/2018/file/1b318124e37af6d74a03501474f44ea1-Paper.pdf
"""
from functools import partial

import jax
import jax.numpy as jnp
import optax
from jax import jit, vmap

from ._sghagame import SGHAGame
from ._utils import _get_AB
from ..._utils import _split_eigenvector


class AlphaSGHAGame(SGHAGame):
    def __init__(self, mode, init_rng, config):
        super(AlphaSGHAGame, self).__init__(mode, init_rng, config)
        """Constructs the experiment.
        Args:
          mode: A string, equivalent to FLAGS.jaxline_mode when running normally.
          init_rng: A `PRNGKey` to use for experiment initialization.
        """
        """Initialization function for a Jaxline experiment."""
        self.grads = jax.jit(
            jax.vmap(
                jax.grad(self._utils),
                in_axes=(0, 1, 0, None, None, None, None),
            )
        )
        self._utils = jax.jit(
            jax.vmap(
                self._utils,
                in_axes=(0, 1, 0, None, None, None, None),
            )
        )

    def _update(self, views, global_step):
        X_i, Y_i = views
        w_grad = self._grad(X_i, Y_i, self._W,self._W, self._weights)
        updates, self._opt_state = self._optimizer.update(-w_grad, self._opt_state)
        self._W = optax.apply_updates(self._W, updates)
        self._W = self._normalize(self._W)
        self._U, self._V = _split_eigenvector(self._W, X_i.shape[1])

    @staticmethod
    def _utils(X_i, Y_i,w, W,weights):
        A, B = _get_AB(X_i, Y_i)
        rewards=w.T@A@w
        penalties=(w@B@W.T) @ ((w @ A @ W.T) * weights)
        return rewards-penalties
