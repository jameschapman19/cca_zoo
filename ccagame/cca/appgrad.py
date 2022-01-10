"""
Appgrad
"""
from functools import partial
from os import environ

import jax
import jax.numpy as jnp
import optax
from jax import jit
import jax.scipy as jsp
from ccagame.cca.utils import mat_pow
from scipy.linalg import sqrtm
from . import CCAExperiment


class AppGrad(CCAExperiment):
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
        whitening_batch_size=None,
        c=None,
        **kwargs
    ):
        super(AppGrad, self).__init__(
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
        self._U = jax.random.normal(self.init_rng, (self.n_components, self.dims[0]))
        self._U /= jnp.linalg.norm(self._U, axis=1, keepdims=True)
        self._V = jax.random.normal(self.init_rng, (self.n_components, self.dims[1]))
        self._V /= jnp.linalg.norm(self._V, axis=1, keepdims=True)
        self._U_tilde = jnp.zeros_like(self._U)
        self._V_tilde = jnp.zeros_like(self._V)
        self._update_with_grads = jax.jit(
            jax.vmap(
                self._update_with_grads,
                in_axes=(0, 0, 0),
            )
        )
        self._optimizer = optax.sgd(learning_rate=learning_rate)
        self._opt_state_x = self._optimizer.init(self._U_tilde)
        self._opt_state_y = self._optimizer.init(self._V_tilde)
        self.c = c
        if self.c is None:
            self.c = 5e-3
        if whitening_batch_size is None:
            whitening_batch_size = 10 * n_components
        self.whitening_data_stream = self._init_data_stream(
            whitening_batch_size, random_state=1
        )

    def _update(self, views, global_step):
        X_i, Y_i = views
        X_iw, Y_iw = next(self.whitening_data_stream)
        x_grads = self._grad(X_i, Y_i, self._V, self._U_tilde, self.c)
        self._U_tilde, self._opt_state_x = self._update_with_grads(
            self._U, x_grads, self._opt_state_x
        )
        self._U = self._U_tilde / jnp.linalg.norm(self._U_tilde, axis=1, keepdims=True)
        y_grads = self._grad(Y_i, X_i, self._U, self._V_tilde, self.c)
        self._V_tilde, self._opt_state_y = self._update_with_grads(
            self._V, y_grads, self._opt_state_y
        )
        self._V = self._V_tilde / jnp.linalg.norm(self._V_tilde, axis=1, keepdims=True)

    @staticmethod
    # @jit
    def _grad(X_i, Y_i, V, U_tilde, c):
        n = X_i.shape[0]
        grads = (X_i.T @ (X_i @ U_tilde.T) - X_i.T @ Y_i @ V.T) / n + c * U_tilde.T
        return grads.T  # grads[:,0]

    @partial(jit, static_argnums=(0))
    def _update_with_grads(self, wi, grads, opt_state):
        updates, opt_state = self._optimizer.update(grads, opt_state)
        wi_new = optax.apply_updates(wi, updates)
        return wi_new, opt_state

    @staticmethod
    # @jit
    def _normalize(X_i, U, c):
        n = X_i.shape[0]
        M = (U @ X_i.T @ X_i @ U.T) / n + c * jnp.eye(U.shape[0])
        return (U.T @ jnp.linalg.inv(sqrtm(M))).T
