from functools import partial

import jax
import jax.numpy as jnp
import optax
from jax import jit, vmap

from ._sgha import SGHA
from ._utils import _get_AB
from ..._utils import _split_eigenvector


class SGHAGame(SGHA):
    def __init__(self, mode, init_rng, config):
        super(SGHAGame, self).__init__(mode, init_rng, config)
        """Constructs the experiment.
        Args:
          mode: A string, equivalent to FLAGS.jaxline_mode when running normally.
          init_rng: A `PRNGKey` to use for experiment initialization.
        """
        """Initialization function for a Jaxline experiment."""
        self._weights = 2*jnp.ones((config.n_components, config.n_components))
        self._weights = self._weights.at[jnp.triu_indices(config.n_components)].set(
            0
        )

    def _update(self, views, global_step):
        X_i, Y_i = views
        w_grad = self._grad(X_i, Y_i, self._W,self._W, self._weights)
        updates, self._opt_state = self._optimizer.update(-w_grad, self._opt_state)
        self._W = optax.apply_updates(self._W, updates)
        self._W = self._normalize(self._W)
        self._U, self._V = _split_eigenvector(self._W, X_i.shape[1])

    @staticmethod
    @jit
    @partial(vmap, in_axes=(None, None, 0, None, 0))
    def _grad(X_i, Y_i,w, W,weights):
        A, B = _get_AB(X_i, Y_i)
        rewards=A@w - B@w*(w @ A @ w)
        penalties=(B@W.T) @ ((w @ A @ W.T) * weights)
        return rewards-penalties
