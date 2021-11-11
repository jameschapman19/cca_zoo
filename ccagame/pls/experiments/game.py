import functools
from os import environ
import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import app, flags
from ccagame.baseexperiment import get_config
from ccagame.pls.experiments import PLSExperiment
from ccagame.utils import data_stream
from datasets.mnist import mnist
from jaxline import platform

CORES = 4
environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={CORES}"
FLAGS = flags.FLAGS


class Game(PLSExperiment):
    def __init__(
        self,
        mode,
        init_rng=None,
        num_devices=1,
        k_per_device=1,
        dims=1,
        data_stream=None,
        whole_batch=False
    ):
        super(Game, self).__init__(
            mode,
            init_rng=init_rng,
            num_devices=num_devices,
            k_per_device=k_per_device,
            dims=dims,
            data_stream=data_stream,
            whole_batch=whole_batch
        )
        """Constructs the experiment.
        Args:
          mode: A string, equivalent to FLAGS.jaxline_mode when running normally.
          init_rng: A `PRNGKey` to use for experiment initialization.
        """
        """Initialization function for a Jaxline experiment."""
        weights = np.eye(self._total_k) * 2 - np.ones((self._total_k, self._total_k))
        weights[np.triu_indices(self._total_k, 1)] = 0
        self._weights = jnp.reshape(weights, [num_devices, k_per_device, self._total_k])
        # generates a key for each device
        keys = jax.random.split(self.local_rng, num_devices)
        # generates weights for each component on each device
        U = jax.pmap(lambda key: jax.random.normal(key, (k_per_device, dims[0])))(keys)
        V = jax.pmap(lambda key: jax.random.normal(key, (k_per_device, dims[1])))(keys)
        # normalizes the weights for each component
        self._U = jax.pmap(lambda U: U / jnp.linalg.norm(U, axis=1, keepdims=True))(U)
        self._V = jax.pmap(lambda V: V / jnp.linalg.norm(V, axis=1, keepdims=True))(V)
        # This line parallelizes over data sending different data to each device
        self._grads_and_update_x = functools.partial(self._grads_and_update, view="x")
        self._grads_and_update_y = functools.partial(self._grads_and_update, view="y")
        self._grads_and_update_x = jax.pmap(
            self._grads_and_update_x,
            in_axes=(0, 0, 0, None, None, 0, 0, 0),
            axis_name="i",
        )
        self._grads_and_update_y = jax.pmap(
            self._grads_and_update_y,
            in_axes=(0, 0, 0, None, None, 0, 0, 0),
            axis_name="i",
        )
        # This parallelizes gradient calcs and updates for eigenvectors within a given device
        self._grads_and_utils = jax.vmap(
            self._grads_and_utils, in_axes=(0, 0, 0, None, None, None, None)
        )
        self._optimizer_x = optax.sgd(learning_rate=1e-5, momentum=0.9, nesterov=True)
        self._opt_state_x = jax.pmap(lambda U: self._optimizer_x.init(U))(self._U)
        self._optimizer_y = optax.sgd(learning_rate=1e-5, momentum=0.9, nesterov=True)
        self._opt_state_y = jax.pmap(lambda V: self._optimizer_y.init(V))(self._V)

    def _update(self, views, global_step):
        X_i, Y_i = views
        X_i = jnp.reshape(X_i, (self._num_devices, -1, self._dims[0]))
        Y_i = jnp.reshape(Y_i, (self._num_devices, -1, self._dims[1]))
        self._local_U = jnp.reshape(self._U, (self._total_k, self._dims[0]))
        self._local_V = jnp.reshape(self._V, (self._total_k, self._dims[1]))
        self._U, self._opt_state_x = self._grads_and_update_x(
            self._U,
            self._V,
            self._weights,
            self._local_U,
            self._local_V,
            X_i,
            Y_i,
            self._opt_state_x,
        )
        self._V, self._opt_state_y = self._grads_and_update_y(
            self._V,
            self._U,
            self._weights,
            self._local_V,
            self._local_U,
            Y_i,
            X_i,
            self._opt_state_y,
        )
        return jnp.reshape(self._U, (self._total_k, self._dims[0])), jnp.reshape(
            self._V, (self._total_k, self._dims[1])
        )

    def _grads_and_update(self, ui, vi, weights, U, V, X, Y, opt_state, view="x"):
        """
        Compute utilities and update directions, psum and apply.
        Args:
        vi: shape (k_per_device,d,), eigenvector to be updated
        weights: shape (k_per_device, k,), mask for penalty coefficients,
        V: shape (k, d), i.e., vectors on rows
        input: shape (N, d), minibatch X_t
        opt_state: optax state
        axis_index_groups: For multi-host parallelism https://jax.readthedocs.io/en/latest/
        _modules/jax/_src/lax/parallel.html
        Returns:
        vi_new: shape (d,), eigenvector to be updated
        opt_state: new optax state
        utilities: shape (1,), utilities
        """
        grads, utilities = self._grads_and_utils(ui, vi, weights, U, V, X, Y)
        # avg_grads = jax.lax.psum(
        #    grads, axis_name='i', axis_index_groups=axis_index_groups)
        ui_new, opt_state = self._update_with_grads(ui, grads, opt_state, view=view)
        return ui_new, opt_state

    def _update_with_grads(self, ui, grads, opt_state, view="x"):
        """Compute and apply updates with optax optimizer.
        Wrap in jax.vmap for k_per_device dimension."""
        if view == "x":
            updates, opt_state = self._optimizer_x.update(grads, opt_state)
        else:
            updates, opt_state = self._optimizer_y.update(grads, opt_state)
        ui_new = optax.apply_updates(ui, updates)
        ui_new = jnp.diag(1 / jnp.linalg.norm(ui_new, axis=1)) @ ui_new
        return ui_new, opt_state

    @staticmethod
    def eg_grads(
        ui: jnp.ndarray, vi, weights: jnp.ndarray, U, V, X: jnp.ndarray, Y: jnp.ndarray
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
        C = X.T @ Y
        penalty_grads = ui @ C @ V.T * U.T
        penalty_grads = penalty_grads @ weights_ij
        grads = C @ vi.T + penalty_grads
        return grads

    @staticmethod
    def utility(ui, weights, U, V, X, Y):
        """Compute Eigengame utilities.
        util: shape (1,), utility for vi
        """
        C = X.T @ Y  # Xvj on row j
        vi_m_vj2 = ui @ C @ V.T ** 2.0
        vj_m_vj = jnp.diag(U @ C @ V.T)
        r_ij = vi_m_vj2 / vj_m_vj
        util = r_ij @ weights
        return util

    @staticmethod
    def _grads_and_utils(ui, vi, weights, U, V, X, Y):
        """Compute utiltiies and update directions ("grads").
        23 Wrap in jax.vmap for k_per_device dimension."""
        utilities = Game.utility(ui, weights, U, V, X, Y)
        grads = Game.eg_grads(ui, vi, weights, U, V, X, Y)
        return grads, utilities


# TO RUN AN EXPERIMENT YOU HAVE TO TINKER HERE A BIT.
if __name__ == "__main__":
    X, _, X_te, _ = mnist()
    input_data_iterator = data_stream(X[:, :400], Y=X[:, 400:], batch_size=None)
    k_per_device = 5
    FLAGS.config = get_config(input_data_iterator, CORES, [X.shape[1],Y.shape[1]],k_per_device=k_per_device)
    flags.mark_flag_as_required("config")
    app.run(functools.partial(platform.main, Game))
