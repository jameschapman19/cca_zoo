"""
Gen-Oja: A Simple and Efficient Algorithm for
Streaming Generalized Eigenvector Computation
https://proceedings.neurips.cc/paper/2018/file/1b318124e37af6d74a03501474f44ea1-Paper.pdf
"""
from functools import partial
from os import environ
import jax
import jax.numpy as jnp
import optax

from ccagame.cca.utils import _get_AB
from . import CCAExperiment
import jax.scipy as jsp
from jax import jit


class SGHA(CCAExperiment):
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
        super(SGHA, self).__init__(
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
        self.W = (
            jax.random.normal(
                self.local_rng, (self.n_components, self.dims[0] + self.dims[1])
            )
        ) / 1000
        self._update_with_grads = jax.jit(
            jax.vmap(
                self._update_with_grads,
                in_axes=(0, 1, 0),
            )
        )
        self._optimizer = optax.sgd(learning_rate=learning_rate)
        self._opt_state = self._optimizer.init(self.W)

    def _update(self, views, global_step):
        X_i, Y_i = views
        w_grad = self._grad(X_i, Y_i, self.W)
        self.W, self._opt_state = self._update_with_grads(
            self.W, w_grad, self._opt_state
        )
        self._U, self._V = self._split_eigenvector(self.W)

    @staticmethod
    @jit
    def _grad(X_i, Y_i, W):
        A, B = _get_AB(X_i, Y_i)
        Y = W @ A @ W.T
        return B @ W.T @ jnp.triu(Y) - A @ W.T

    @partial(jit, static_argnums=(0))
    def _update_with_grads(self, wi, grads, opt_state):
        # we have gradient of utilities so we negate for gradient descent
        updates, opt_state = self._optimizer.update(-grads, opt_state)
        wi_new = optax.apply_updates(wi, updates)
        return wi_new, opt_state

    @partial(jit, static_argnums=(0))
    def _split_eigenvector(self,V):
        return self.W[:, : self.dims[0]], self.W[:, self.dims[0] :]
