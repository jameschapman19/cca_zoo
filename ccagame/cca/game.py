from functools import partial
from os import environ
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import jit
from . import CCAExperiment


class Game(CCAExperiment):
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
        super(Game, self).__init__(
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
        k_per_device = int(n_components / num_devices)
        weights = np.eye(self.n_components) * 2 - np.ones(
            (self.n_components, self.n_components)
        )
        weights[np.triu_indices(self.n_components, 1)] = 0
        self._weights = jnp.reshape(
            weights, [num_devices, k_per_device, self.n_components]
        )
        # generates a key for each device
        keys = jax.random.split(self.local_rng, num_devices)
        # generates weights for each component on each device
        self._U = jax.pmap(lambda key: jax.random.normal(key, (k_per_device, dims[0]))/10000)(keys)
        self._V = jax.pmap(lambda key: jax.random.normal(key, (k_per_device, dims[1]))/10000)(keys)
        # This line parallelizes over data sending different data to each device
        self._update_with_grads = jax.pmap(
            jax.vmap(
                self._update_with_grads,
                in_axes=(0, 0, 0),
            ),
            in_axes=(0, 0, 0),
            axis_name="i",
        )
        # This parallelizes gradient calcs and updates for eigenvectors within a given device
        self._grads = jax.pmap(
            jax.vmap(
                self._grads,
                in_axes=(0, 0, 0, None, None, None),
            ),
            in_axes=(0, 0, 0, None, 0, 0),
            axis_name="i",
        )
        self._optimizer = optax.sgd(
            learning_rate=learning_rate, momentum=momentum, nesterov=nesterov
        )
        self._opt_state_x = jax.pmap(lambda U: self._optimizer.init(U))(self._U)
        self._opt_state_y = jax.pmap(lambda V: self._optimizer.init(V))(self._V)

    # @partial(jit, static_argnums=(0))
    def _update(self, views, global_step):
        X_i, Y_i = views
        X_i = jnp.reshape(X_i, (self.num_devices, -1, self.dims[0]))
        Y_i = jnp.reshape(Y_i, (self.num_devices, -1, self.dims[1]))
        self._local_U = jnp.reshape(self._U, (self.n_components, self.dims[0]))
        self._local_V = jnp.reshape(self._V, (self.n_components, self.dims[1]))
        T = X_i @ self._local_U.T + Y_i @ self._local_V.T
        T = T / jnp.linalg.norm(T, axis=0)
        local_T = T
        T = jnp.reshape(T, (self.num_devices, -1, T.shape[1]))
        grads_x = self._grads(self._U, T, self._weights, self._local_U, local_T, X_i)
        grads_y = self._grads(self._V, T, self._weights, self._local_V, local_T, Y_i)
        self._U, self._opt_state_x = self._update_with_grads(
            self._U, grads_x, self._opt_state_x
        )
        self._V, self._opt_state_y = self._update_with_grads(
            self._V, grads_y, self._opt_state_y
        )

    @staticmethod
    @jit
    def _grads(ui, ti, weights, U, T, X):
        weights_ij = (jnp.sign(weights + 0.5) - 1.0) / 2.0  # maps -1 to -1 else to 0
        reward = X.T @ ti
        variance_penalty = X.T @ X @ ui
        penalty_grads = ui @ X.T @ T * U.T
        penalty_grads = penalty_grads @ weights_ij
        grads = reward - variance_penalty + penalty_grads
        return grads/X.shape[0]

    @partial(jit, static_argnums=(0))
    def _update_with_grads(self, ui, grads, opt_state):
        updates, opt_state = self._optimizer.update(grads, opt_state)
        ui_new = optax.apply_updates(ui, updates)
        return ui_new, opt_state
