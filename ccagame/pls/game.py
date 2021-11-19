from os import environ
import jax
import jax.numpy as jnp
import numpy as np
import optax
from . import PLSExperiment


class Game(PLSExperiment):
    def __init__(
        self,
        mode,
        init_rng=None,
        num_devices=1,
        n_components=1,
        dims=None,
        data=None,
        learning_rate=1e-3,
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
        U = jax.pmap(lambda key: jax.random.normal(key, (k_per_device, dims[0])))(keys)
        V = jax.pmap(lambda key: jax.random.normal(key, (k_per_device, dims[1])))(keys)
        # normalizes the weights for each component
        self._U = jax.pmap(lambda U: U / jnp.linalg.norm(U, axis=1, keepdims=True))(U)
        self._V = jax.pmap(lambda V: V / jnp.linalg.norm(V, axis=1, keepdims=True))(V)
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
                in_axes=(0, 0, 0, None, None, None, None),
            ),
            in_axes=(0, 0, 0, None, None, 0, 0),
            axis_name="i",
        )
        self._optimizer = optax.sgd(
            learning_rate=learning_rate, momentum=momentum, nesterov=nesterov
        )
        self._opt_state_x = jax.pmap(lambda U: self._optimizer.init(U))(self._U)
        self._opt_state_y = jax.pmap(lambda V: self._optimizer.init(V))(self._V)

    def _update(self, views, global_step):
        X_i, Y_i = views
        X_i = jnp.reshape(X_i, (self.num_devices, -1, self.dims[0]))
        Y_i = jnp.reshape(Y_i, (self.num_devices, -1, self.dims[1]))
        self._local_U = jnp.reshape(self._U, (self.n_components, self.dims[0]))
        self._local_V = jnp.reshape(self._V, (self.n_components, self.dims[1]))
        grads_x = self._grads(
            self._U, self._V, self._weights, self._local_U, self._local_V, X_i, Y_i
        )
        grads_y = self._grads(
            self._V, self._U, self._weights, self._local_V, self._local_U, Y_i, X_i
        )
        self._U, self._opt_state_x = self._update_with_grads(
            self._U, grads_x, self._opt_state_x
        )
        self._V, self._opt_state_y = self._update_with_grads(
            self._V, grads_y, self._opt_state_y
        )

    @staticmethod
    def _grads(ui, vi, weights, U, V, X, Y):
        """Compute utiltiies and update directions ("grads").
        23 Wrap in jax.vmap for k_per_device dimension."""
        Z = Y @ V.T
        zi = Y @ vi
        grads = Game.eg_grads(ui, zi, weights, U, Z, X)
        return grads

    def _update_with_grads(self, ui, grads, opt_state):
        """Compute and apply updates with optax optimizer.
        Wrap in jax.vmap for k_per_device dimension."""
        updates, opt_state = self._optimizer.update(grads, opt_state)
        ui_new = optax.apply_updates(ui, updates)
        ui_new /= jnp.linalg.norm(ui_new)
        return ui_new, opt_state

    @staticmethod
    def eg_grads(
        ui: jnp.ndarray, zi, weights: jnp.ndarray, U, Z, X: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Args:
        vi: shape (d,), eigenvector to be updated
        weights: shape (k,), mask for penalty coefficients,
        eigs: shape (k, d), i.e., vectors on rows
        data: shape (N, d), minibatch X_t
        Returns:
        grads: shape (d,), gradient for vi
        """
        weights_ij = (jnp.sign(weights + 0.5) - 1.0) / 2.0  # maps -1 to -1 else to 0
        penalty_grads = ui @ X.T @ Z * U.T
        penalty_grads = penalty_grads @ weights_ij
        grads = X.T @ zi + penalty_grads
        return grads

    @staticmethod
    def utility(ui, weights, U, Z, X):
        """Compute Eigengame utilities.
        util: shape (1,), utility for vi
        """
        vi_m_vj2 = ui @ X.T @ Z ** 2.0
        vj_m_vj = jnp.diag(U @ X.T @ Z)
        r_ij = vi_m_vj2 / vj_m_vj
        util = r_ij @ weights
        return util
