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

from ccagame.cca.utils import _get_AB, _gram_schmidt

from . import CCAExperiment
import jax.scipy as jsp
from jax import jit


class GenOja(CCAExperiment):
    def __init__(
        self,
        mode,
        init_rng=None,
        num_devices=1,
        n_components=1,
        dims=None,
        data=None,
        alpha=1 / 871,
        beta0=1000,
        batch_size=0,
        **kwargs
    ):
        super(GenOja, self).__init__(
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
            / 10000
        )
        self.V = jax.random.normal(
            self.local_rng, (self.n_components, self.dims[0] + self.dims[1])
        )
        self.V = self.V / jnp.linalg.norm(self.V, keepdims=True, axis=1)

        self._grads = jax.jit(jax.vmap(
                    self._grads,
                    in_axes=(None,None,0,0),
                ))
        self._update_with_grads_ls = jax.jit(
            jax.vmap(
                self._update_with_grads_ls,
                in_axes=(0, 0, 0),
            )
        )
        self._update_with_grads_oja = jax.jit(
            jax.vmap(
                self._update_with_grads_oja,
                in_axes=(0, 0, 0),
            )
        )

        self._optimizer_ls = optax.sgd(learning_rate=alpha)
        self._optimizer_oja = optax.sgd(learning_rate=beta0)
        self._opt_state_ls = self._optimizer_ls.init(self.W)
        self._opt_state_oja = self._optimizer_oja.init(self.V)

    def _update(self, views, global_step):
        X_i, Y_i = views
        w_grad = self._grads(X_i, Y_i, self.W, self.V)
        self.W, self._opt_state_ls = self._update_with_grads_ls(
            self.W, w_grad, self._opt_state_ls
        )
        self.V, self._opt_state_oja = self._update_with_grads_oja(
            self.V, self.W / (global_step + 1), self._opt_state_oja
        )
        self.V = _gram_schmidt(self.V, _get_AB(X_i, Y_i)[1])
        self._U, self._V = self._split_eigenvector(self.V)

    @staticmethod
    def _grads(X_i, Y_i, W, V):
        A, B = _get_AB(X_i, Y_i)
        return B @ W - A @ V

    @partial(jit, static_argnums=(0))
    def _update_with_grads_ls(self, wi, grads, opt_state):
        # we have gradient of utilities so we negate for gradient descent
        updates, opt_state = self._optimizer_ls.update(grads, opt_state)
        wi_new = optax.apply_updates(wi, updates)
        return wi_new, opt_state

    @partial(jit, static_argnums=(0))
    def _update_with_grads_oja(self, vi, grads, opt_state):
        # we have gradient of utilities so we negate for gradient descent
        updates, opt_state = self._optimizer_oja.update(-grads, opt_state)
        vi_new = optax.apply_updates(vi, updates)
        return vi_new, opt_state

    @partial(jit, static_argnums=(0))
    def _split_eigenvector(self,V):
        return self.W[:, : self.dims[0]], self.W[:, self.dims[0] :]
