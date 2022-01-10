from functools import partial
from os import environ

import jax
import jax.numpy as jnp
import optax
from ccagame.cca import CCAExperiment
from jax import jit


class AltGame(CCAExperiment):
    NON_BROADCAST_CHECKPOINT_ATTRS = {"_U": "U", "_V": "V"}

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
        alpha=True,
        **kwargs
    ):
        super(AltGame, self).__init__(
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
        self._weights = jnp.ones((self.n_components, self.n_components)) - jnp.eye(
            self.n_components
        )
        self._weights = self._weights.at[jnp.triu_indices(self.n_components, 1)].set(0)
        # generates weights for each component on each device
        self._U = jax.random.normal(self.init_rng, (self.n_components, self.dims[0]))
        self._U /= jnp.linalg.norm(self._U, axis=1, keepdims=True)
        self._V = jax.random.normal(self.init_rng, (self.n_components, self.dims[1]))
        self._V /= jnp.linalg.norm(self._V, axis=1, keepdims=True)
        # This parallelizes gradient calcs and updates for eigenvectors within a given device
        self._grads = jax.jit(
            jax.vmap(self._grads, in_axes=(1, 0, None, 1, None, None))
        )
        self._update_with_grads = jax.jit(
            jax.vmap(
                self._update_with_grads,
                in_axes=(0, 0, 0),
            )
        )
        self._optimizer = optax.sgd(learning_rate=learning_rate)
        self._opt_state_x = self._optimizer.init(self._U)
        self._opt_state_y = self._optimizer.init(self._V)
        self.learning_rate = learning_rate
        self.auxiliary_data = self._init_data_stream(self.batch_size, random_state=1)

    def _update(self, views, global_step):
        X_i, Y_i = views
        X_i_aux, Y_i_aux = next(self.auxiliary_data)  # jnp.corrcoef(Zx,Zy,rowvar=False)
        Zx, Zy, T = self._get_target(X_i, Y_i, self._U, self._V)#T.T@T
        Zx_aux, Zy_aux, T_aux = self._get_target(X_i_aux, Y_i_aux, self._U, self._V)
        grads_x = self._grads(Zx, self._weights, X_i, T, T, Zx)
        grads_y = self._grads(Zy, self._weights, Y_i, T, T, Zy)
        self._U, self._opt_state_x = self._update_with_grads(
            self._U, grads_x, self._opt_state_x
        )
        self._V, self._opt_state_y = self._update_with_grads(
            self._V, grads_y, self._opt_state_y
        )

    @staticmethod
    def _grads(zi, weights, X, Ti, T, Z):
        n = X.shape[0]
        rewards = X.T @ (Ti - zi)
        covariance = -((Ti @ T) * (X.T @ (T - Z))) @ weights
        grads = rewards + covariance
        return grads/n

    @staticmethod
    def _utils(ui, weights, X, Ti, T):
        raise NotImplementedError

    @staticmethod
    @jit
    def _get_target(X, Y, U, V):
        Zx = X @ U.T
        Zy = Y @ V.T
        T = Zx + Zy
        T /= jnp.linalg.norm(T, axis=0, keepdims=True)
        return Zx, Zy, T

    @partial(jit, static_argnums=(0))
    def _update_with_grads(self, ui, grads, opt_state):
        # we have gradient of utilities so we negate for gradient descent
        updates, opt_state = self._optimizer.update(-grads, opt_state)
        ui_new = optax.apply_updates(ui, updates)
        # ui_new /= jnp.linalg.norm(ui_new, keepdims=True)
        return ui_new, opt_state
