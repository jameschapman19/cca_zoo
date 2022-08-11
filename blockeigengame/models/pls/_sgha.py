
from functools import partial

import jax
import jax.numpy as jnp
import optax
from jax import jit, vmap

from ._plsmixin import _PLSMixin
from ._utils import _get_AB
from .._baseexperiment import _BaseExperiment
from ..._utils import _split_eigenvector


class SGHA(_PLSMixin, _BaseExperiment):
    def __init__(self, mode, init_rng, config):
        super(SGHA, self).__init__(mode, init_rng, config)
        """Constructs the experiment.
        Args:
          mode: A string, equivalent to FLAGS.jaxline_mode when running normally.
          init_rng: A `PRNGKey` to use for experiment initialization.
        """
        """Initialization function for a Jaxline experiment."""

    def _init_train(self):
        self._init_ground_truth()
        views = next(self._train_input)
        self._W = jax.random.normal(
            self.init_rng,
            (self.config.n_components, views[0].shape[1] + views[1].shape[1]),
        )
        self._W /= jnp.linalg.norm(self._W, axis=1, keepdims=True)
        self._optimizer = optax.sgd(learning_rate=self.config.learning_rate)
        self._opt_state = self._optimizer.init(self._W)

    def _update(self, views, global_step):
        X_i, Y_i = views
        w_grad = self._grad(X_i, Y_i, self._W)
        updates, self._opt_state = self._optimizer.update(-w_grad, self._opt_state)
        self._W = optax.apply_updates(self._W, updates)
        self._W = self._normalize(self._W)
        self._U, self._V = _split_eigenvector(self._W, X_i.shape[1])

    @staticmethod
    @jit
    def _grad(X_i, Y_i, W):
        A, B = _get_AB(X_i, Y_i)
        Y = W @ A @ W.T
        return W@A - Y@W @ B

