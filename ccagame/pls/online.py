from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import optax
from jax import jit

from . import PLSExperiment


def logm(M):
    # TODO
    pass


def expm(M):
    # TODO
    pass


class Online(PLSExperiment):
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
        **kwargs
    ):
        super(Online, self).__init__(
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
        self._U = jax.random.normal(self.local_rng, (self.n_components, self.dims[0]))
        self._U /= jnp.linalg.norm(self._U, axis=1, keepdims=True)
        self._V = jax.random.normal(self.local_rng, (self.n_components, self.dims[1]))
        self._V /= jnp.linalg.norm(self._V, axis=1, keepdims=True)
        self._M = jax.random.normal(
            self.local_rng, (self.dims[0] + self.dims[1], self.dims[0] + self.dims[1])
        )
        self._optimizer = optax.sgd(
            learning_rate=learning_rate, momentum=momentum, nesterov=nesterov
        )
        self._opt_state = self._optimizer.init(self._M)

    @partial(jit, static_argnums=(0))
    def _update(self, views, global_step):
        X_i, Y_i = views
        Z_t = 0.5 * jnp.hstack((X_i, Y_i)).T @ jnp.hstack(
            (X_i, Y_i)
        ) - 0.5 * jnp.hstack((X_i, -Y_i)).T @ jnp.hstack((X_i, -Y_i))
        updates, self._opt_state = self._optimizer.update(Z_t, self._opt_state)
        M = jsp.linalg.expm(optax.apply_updates(jsp.linalg.logm(self._M), updates))
        # set largest d-k eigenvalues to 1/d-k and remaining to satisfy normalization
        self._U, S, V = jnp.linalg.svd(M)
        d = self.dims[0] + self.dims[1]
        scaled = (
            self.n_components
            / d
            * S[-self.n_components :]
            / S[-self.n_components :].sum()
        )
        S = jnp.stack(
            jnp.ones((d - self.n_components)) / (d + self.n_components), scaled
        )  # jnp.sum(scaled.sum()+(jnp.ones((d-self.n_components))/(d+self.n_components)).sum())
        self._M = self._U @ S @ self._V.T
        self._V = V.T
