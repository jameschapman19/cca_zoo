import functools
from os import environ

import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import app, flags
from ccagame.baseexperiment import BaseExperiment, get_config
from ccagame.utils import data_stream
from datasets.mnist import mnist
from jax._src.random import PRNGKey
from jaxline import platform

CORES = 2
environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={CORES}"
FLAGS = flags.FLAGS


class GameExperiment(BaseExperiment):
    def __init__(
        self,
        mode,
        init_rng=None,
        num_devices=1,
        k_per_device=1,
        dims=1,
        axis_index_groups=None,
        data_stream=None,
    ):
        super(GameExperiment, self).__init__(
            mode,
            init_rng=init_rng,
            num_devices=num_devices,
            k_per_device=k_per_device,
            dims=dims,
            axis_index_groups=axis_index_groups,
            data_stream=data_stream,
        )
        """
        Constructs the experiment.
        Args:
          mode: A string, equivalent to FLAGS.jaxline_mode when running normally.
          init_rng: A `PRNGKey` to use for experiment initialization.
        """
        """
        Initialization function for a Jaxline experiment.
        """
        self._total_k = k_per_device * num_devices
        self._dims = dims
        self.data_stream = data_stream
        weights = np.eye(self._total_k) * 2 - np.ones((self._total_k, self._total_k))
        weights[np.triu_indices(self._total_k, 1)] = 0
        self._weights = jnp.reshape(weights, [num_devices, k_per_device, self._total_k])
        local_rng = jax.random.fold_in(PRNGKey(123), jax.host_id())
        # generates a key for each device
        keys = jax.random.split(local_rng, num_devices)
        # generates weights for each component on each device
        V = jax.pmap(lambda key: jax.random.normal(key, (k_per_device, dims)))(keys)
        # normalizes the weights for each component
        self._V = jax.pmap(lambda V: V / jnp.linalg.norm(V, axis=1, keepdims=True))(V)
        # Define parallel update function. If k_per_device is not None, wrap individual functions with vmap here.
        # This ignores multi-host parallelism see https://jax.readthedocs.io/en/latest/_modules/jax/_src/lax/parallel.html
        self._partial_grad_update = functools.partial(
            self._grads_and_update, axis_groups=None
        )
        # This line parallelizes over data sending different data to each device
        self._par_grad_update = jax.pmap(
            self._grads_and_update, in_axes=(0, 0, None, 0, 0, 0), axis_name="i"
        )
        # This parallelizes gradient calcs and updates for eigenvectors within a given device
        self._grads_and_utils = jax.vmap(
            self._grads_and_utils, in_axes=(0, 0, None, None)
        )
        # self._update_with_grads = jax.vmap(self._update_with_grads, in_axes=(0, 0, None))
        self._optimizer = optax.sgd(learning_rate=1e-2, momentum=0.9, nesterov=True)
        self._opt_state = self._optimizer.init(self._V)

    def _update(self, inputs, global_step):
        inputs = jnp.reshape(inputs, (self._V.shape[0], -1, inputs.shape[-1]))
        self._local_V = jnp.reshape(self._V, (self._total_k, self._dims))
        self._V, self._opt_state, utilities = self._par_grad_update(
            self._V, self._weights, self._local_V, inputs, self._opt_state, global_step
        )
        return {"TV": self.TV(inputs, self._V)}

    def TV(self, X, V):
        X = jnp.reshape(X, (-1, X.shape[-1]))
        V = jnp.reshape(V, (-1, X.shape[-1]))
        dof = X.shape[0]
        U = X @ V.T
        jnp.corrcoef(U.T)
        jnp.linalg.svd(X.T @ X)[1]
        return jnp.linalg.svd(U.T @ U)[1].sum() / dof

    def _grads_and_update(self, vi, weights, V, input, opt_state, axis_index_groups):
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
        grads, utilities = self._grads_and_utils(vi, weights, V, input)
        # avg_grads = jax.lax.psum(
        #    grads, axis_name='i', axis_index_groups=axis_index_groups)
        vi_new, opt_state = self._update_with_grads(vi, grads, opt_state)
        return vi_new, opt_state, utilities

    def _update_with_grads(self, vi, grads, opt_state):
        """Compute and apply updates with optax optimizer.
        Wrap in jax.vmap for k_per_device dimension."""
        updates, opt_state = self._optimizer.update(grads, opt_state)
        vi_new = optax.apply_updates(vi, updates)
        vi_new = jnp.diag(1 / jnp.linalg.norm(vi_new, axis=1)) @ vi_new
        return vi_new, opt_state

    @staticmethod
    def eg_grads(
        vi: jnp.ndarray, weights: jnp.ndarray, eigs: jnp.ndarray, data: jnp.ndarray
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
        data_vi = data @ vi
        data_eigs = (data @ eigs.T).T  # Xvj on row j
        vi_m_vj = data_eigs @ data_vi
        penalty_grads = vi_m_vj * eigs.T
        penalty_grads = penalty_grads @ weights_ij
        grads = data.T @ data_vi + penalty_grads
        return grads

    @staticmethod
    def utility(vi, weights, eigs, data):
        """Compute Eigengame utilities.
        util: shape (1,), utility for vi
        """
        data_vi = data @ vi
        data_eigs = (data @ eigs.T).T  # Xvj on row j
        vi_m_vj2 = data_eigs @ data_vi ** 2.0
        vj_m_vj = jnp.sum(data_eigs * data_eigs, axis=1)
        r_ij = vi_m_vj2 / vj_m_vj
        util = r_ij @ weights
        return util

    @staticmethod
    def _grads_and_utils(vi, weights, V, inputs):
        """Compute utiltiies and update directions ("grads").
        23 Wrap in jax.vmap for k_per_device dimension."""
        utilities = GameExperiment.utility(vi, weights, V, inputs)
        grads = GameExperiment.eg_grads(vi, weights, V, inputs)
        return grads, utilities


# TO RUN AN EXPERIMENT YOU HAVE TO TINKER HERE A BIT.
if __name__ == "__main__":
    X, _, X_te, _ = mnist()
    input_data_iterator = data_stream(X, Y=None, batch_size=None)
    FLAGS.config = get_config(input_data_iterator, CORES, X.shape[1])
    flags.mark_flag_as_required("config")
    app.run(functools.partial(platform.main, GameExperiment))
