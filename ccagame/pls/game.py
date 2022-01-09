from functools import partial
from os import environ

import jax
import jax.numpy as jnp
import optax
from jax import jit

from . import PLSExperiment


class Game(PLSExperiment):
    NON_BROADCAST_CHECKPOINT_ATTRS = {"_U": "U", "_V": "V"}

    def __init__(
        self,
        mode,
        init_rng=None,
        num_devices=1,
        n_components=1,
        data=None,
        learning_rate=1e-6,
        momentum=0.9,
        nesterov=True,
        batch_size=0,
        alpha=False,
        **kwargs
    ):
        super(Game, self).__init__(
            mode,
            init_rng=init_rng,
            num_devices=num_devices,
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
        self._weights = jnp.ones((self.n_components, self.n_components)) - jnp.eye(
            self.n_components
        )
        self._weights = self._weights.at[jnp.triu_indices(self.n_components, 1)].set(0)
        # generates weights for each component on each device
        self._U = jax.random.normal(self.local_rng, (self.n_components, self.dims[0]))
        self._U /= jnp.linalg.norm(self._U, axis=1, keepdims=True)
        self._V = jax.random.normal(self.local_rng, (self.n_components, self.dims[1]))
        self._V /= jnp.linalg.norm(self._V, axis=1, keepdims=True)
        # This parallelizes gradient calcs and updates for eigenvectors within a given device
        self._grads = jax.jit(
            jax.vmap(
                self._grads,
                in_axes=(1, 1, 0, None, None, None, None),
            )
        )
        self._alpha_grads = jax.jit(
            jax.vmap(
                jax.grad(self._utils),
                in_axes=(0, 1, 0, None, None, None),
            )
        )
        self._update_with_grads = jax.jit(
            jax.vmap(
                self._update_with_grads,
                in_axes=(0, 0, 0),
            )
        )
        self._optimizer = optax.sgd(
            learning_rate=learning_rate, momentum=momentum, nesterov=nesterov
        )
        self._opt_state_x = self._optimizer.init(self._U)
        self._opt_state_y = self._optimizer.init(self._V)
        self.learning_rate = learning_rate
        self.alpha = alpha

    def _update(self, views, global_step):
        (
            X_i,
            Y_i,
        ) = views
        Zx, Zy = self._get_target(X_i, Y_i, self._U, self._V)#jnp.corrcoef(Zx,Zy,rowvar=False)
        if self.alpha:
            grads_x = self._alpha_grads(self._U, Zy, self._weights, X_i, Zx, Zy)
            grads_y = self._alpha_grads(self._V, Zx, self._weights, Y_i, Zy, Zx)
        else:
            grads_x = self._grads(Zx, Zy, self._weights, X_i, self._U, Zx, Zy)
            grads_y = self._grads(Zy, Zx, self._weights, Y_i, self._V, Zy, Zx)
        self._U, self._opt_state_x = self._update_with_grads(
            self._U, grads_x, self._opt_state_x
        )
        self._V, self._opt_state_y = self._update_with_grads(
            self._V, grads_y, self._opt_state_y
        )

    @staticmethod
    def _grads(zx, zy, weights, X, U, Zx, Zy):
        rewards = X.T @ zy
        covariance = -((zx.T @ Zy) * U.T) @ weights
        grads = rewards + covariance
        return grads / zy.shape[0]

    @partial(jit, static_argnums=(0))
    def _update_with_grads(self, ui, grads, opt_state):
        """Compute and apply updates with optax optimizer.
        Wrap in jax.vmap for k_per_device dimension."""
        updates, opt_state = self._optimizer.update(-grads, opt_state)
        ui_new = optax.apply_updates(ui, updates)
        ui_new /= jnp.linalg.norm(ui_new, keepdims=True)
        return ui_new, opt_state

    @staticmethod
    @jit
    def _get_target(X, Y, U, V):
        Zx = X @ U.T
        Zy = Y @ V.T
        return Zx, Zy

    @staticmethod
    def _utils(ux, zy, weights, X, Zx, Zy):
        zx = X @ ux
        rewards = zx @ zy
        covariance = -((zx @ Zy) ** 2) / jnp.diag(Zx.T @ Zy) @ weights
        grads = rewards + covariance
        return grads / zy.shape[0]
