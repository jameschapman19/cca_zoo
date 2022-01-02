from functools import partial
from os import environ

import jax
import jax.numpy as jnp
import optax
from jax import jit

from . import PLSExperiment


class Game(PLSExperiment):
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
        self._U = (1 / jnp.linalg.norm(self._U, axis=1) * self._U.T).T
        self._V = jax.random.normal(self.local_rng, (self.n_components, self.dims[1]))
        self._V = (1 / jnp.linalg.norm(self._V, axis=1) * self._V.T).T
        # This parallelizes gradient calcs and updates for eigenvectors within a given device
        self._alpha_grads = jax.jit(
                jax.vmap(
                    jax.grad(self._utils),
                    in_axes=(0, 1, 0, None, None),
                )
            )
        self._mu_grads = jax.jit(
            jax.vmap(
                self._grads,
                in_axes=(1, 1, 1, 0, None, None, None),
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
        self.auxiliary_data = self._init_data_stream(
        random_state=1
        )

    def _update(self, views, global_step):
        (
            X_i,
            Y_i,
        ) = views
        X_aux_i,Y_aux_i= next(self.auxiliary_data)
        Zx, Zy = self._get_target(X_i, Y_i, self._U, self._V)
        Zx_aux, Zy_aux = self._get_target(X_aux_i, Y_aux_i, self._U, self._V)
        if self.alpha:
            grads_x = self._alpha_grads(self._U, Zy, self._weights, X_i, Zy)
            grads_y = self._alpha_grads(self._V, Zx, self._weights, Y_i, Zx)
        else:
            grads_x = self._mu_grads(Zx, Zy,Zy_aux,self._weights, X_aux_i, Zy,Zy_aux)
            grads_y = self._mu_grads(Zy, Zx,Zx_aux,self._weights, Y_aux_i, Zx,Zx_aux)
        self._U, self._opt_state_x = self._update_with_grads(
            self._U, grads_x, self._opt_state_x
        )
        self._V, self._opt_state_y = self._update_with_grads(
            self._V, grads_y, self._opt_state_y
        )

    @staticmethod
    def _grads(zx, zy, zy_aux, weights, X_aux, Zy, Zy_aux):
        rewards = jnp.dot(zx, zy) * (X_aux.T @ zy_aux)
        covariance = -((zx @ Zy) * (X_aux.T @ Zy_aux)) @ weights
        grads = rewards + covariance
        return grads / zx.shape[0]**2

    @staticmethod
    def _utils(ui, zy, weights, X, Zy):
        """Compute utiltiies and update directions ("grads").
        23 Wrap in jax.vmap for k_per_device dimension."""
        zx = X @ ui
        rewards = jnp.dot(zx, zy)**2
        covariance = -((zx @ Zy) ** 2) @ weights
        grads = rewards + covariance
        return grads / X.shape[0]**2

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
