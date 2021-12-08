from os import environ
import jax
import jax.numpy as jnp
import numpy as np
import optax
from .pcaexperiment import PCAExperiment
from jax import jit
from functools import partial

class Game(PCAExperiment):
    def __init__(
        self,
        mode,
        init_rng=None,
        num_devices=1,
        n_components=1,
        data=None,
        batch_size=0,
        learning_rate=1e-3,
        momentum=0.9,
        nesterov=True,
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
        """
        Constructs the experiment.
        Args:
          mode: A string, equivalent to FLAGS.jaxline_mode when running normally.
          init_rng: A `PRNGKey` to use for experiment initialization.
        """
        """
        Initialization function for a Jaxline experiment.
        """
        k_per_device=int(self.n_components/self.num_devices)
        weights = np.eye(self.n_components) - np.ones((self.n_components, self.n_components))
        weights[np.triu_indices(self.n_components, 1)] = 0
        self._weights = jnp.reshape(weights, [num_devices, k_per_device, self.n_components])
        # generates a key for each device
        keys = jax.random.split(self.local_rng, num_devices)
        # generates weights for each component on each device
        V = jax.pmap(lambda key: jax.random.normal(key, (k_per_device, self.dims)))(keys)
        # normalizes the weights for each component
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
            in_axes=(0, 0,None,None),
            ),
            in_axes=(0, 0, None,0),
            axis_name="i"
        )
        # self._update_with_grads = jax.vmap(self._update_with_grads, in_axes=(0, 0, None))
        self._optimizer = optax.sgd(learning_rate=learning_rate, momentum=momentum, nesterov=nesterov)
        self._opt_state = jax.pmap(lambda V: self._optimizer.init(V))(self._V)

    def _update(self, inputs, global_step):
        inputs = jnp.reshape(inputs, (self.num_devices, -1, self.dims))
        self._local_V = jnp.reshape(self._V, (self.n_components, self.dims))
        grads = self._grads(self._V, self._weights, self._local_V, inputs)
        self._V, self._opt_state_y = self._update_with_grads(self._V, grads, self._opt_state)

    @partial(jit, static_argnums=(0))
    def _update_with_grads(self, vi, grads, opt_state):
        """Compute and apply updates with optax optimizer.
        Wrap in jax.vmap for k_per_device dimension."""
        updates, opt_state = self._optimizer.update(-grads, opt_state)
        vi_new = optax.apply_updates(vi, updates)
        vi_new /= jnp.linalg.norm(vi_new)
        return vi_new, opt_state

    @staticmethod
    @jit
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
    @jit
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
    @jit
    def _grads(vi, weights, V, inputs):
        """Compute utiltiies and update directions ("grads").
        23 Wrap in jax.vmap for k_per_device dimension."""
        #utilities = Game.utility(vi, weights, V, inputs)
        grads = Game.eg_grads(vi, weights, V, inputs)
        return grads
