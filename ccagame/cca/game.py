from functools import partial
from os import environ
import jax
import jax.numpy as jnp
import optax
from jax import jit
from ccagame.cca import CCAExperiment


class Game(CCAExperiment):
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
        self._weights = jnp.ones((self.n_components, self.n_components)) - jnp.eye(
            self.n_components
        )
        self._weights = self._weights.at[jnp.triu_indices(self.n_components, 1)].set(0)
        # generates weights for each component on each device
        self._U = (
            jax.random.normal(self.local_rng, (self.n_components, self.dims[0])) / 10000
        )
        self._V = (
            jax.random.normal(self.local_rng, (self.n_components, self.dims[1])) / 10000
        )
        # This parallelizes gradient calcs and updates for eigenvectors within a given device
        self._grads = jax.jit(
            jax.vmap(
                self._grads,
                in_axes=(1,1, 0, None, 1,None, None)
            )
        )
        self._alpha_grads = jax.jit(
            jax.vmap(
                jax.grad(self._utils),
                in_axes=(0,1, 0, None, 1,None, None)
            )
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
        self.alpha=alpha
        self.auxiliary_data = self._init_data_stream(self.batch_size,random_state=1)

    def _update(self, views, global_step):
        X_i, Y_i = views
        X_i_aux,Y_i_aux= next(self.auxiliary_data)
        Zx, Zy, T = self._get_target(X_i, Y_i, self._U, self._V)
        Zx_aux, Zy_aux, T_aux = self._get_target(X_i_aux, Y_i_aux, self._U, self._V)
        if self.alpha:
            grads_x = self._alpha_grads(self._U,Zx_aux, self._weights, X_i, T, T,T_aux)
            grads_y = self._alpha_grads(self._V,Zy_aux, self._weights, Y_i, T, T,T_aux)
        else:
            grads_x = self._grads(Zx,Zx_aux, self._weights, X_i, T, T,T_aux)
            grads_y = self._grads(Zy,Zy_aux, self._weights, Y_i, T, T,T_aux)
        self._U, self._opt_state_x = self._update_with_grads(
            self._U, grads_x, self._opt_state_x
        )
        self._V, self._opt_state_y = self._update_with_grads(
            self._V, grads_y, self._opt_state_y
        )

    @staticmethod
    def _grads(zi,zi_aux, weights, X, Ti,T,T_aux):
        rewards = X.T @ (Ti - zi)
        covariance = -((zi_aux @ T_aux) * (X.T @ T)) @ weights
        grads = rewards + covariance
        return grads / X.shape[0]

    @staticmethod
    def _utils(ui,zi_aux, weights, X, Ti,T,T_aux):
        zi=X@ui
        rewards = jnp.linalg.norm(zi-Ti)**2
        covariance = -(zi_aux @ T_aux)**2 @ weights
        grads = rewards + covariance
        return grads / X.shape[0]

    @staticmethod
    @jit
    def _get_target(X, Y, U, V):
        Zx = X @ U.T
        Zy = Y @ V.T
        T = Zx + Zy
        T /= jnp.linalg.norm(T, axis=0)
        return Zx, Zy, T

    @partial(jit, static_argnums=(0))
    def _update_with_grads(self, ui, grads, opt_state):
        # we have gradient of utilities so we negate for gradient descent
        updates, opt_state = self._optimizer.update(-grads, opt_state)
        ui_new = optax.apply_updates(ui, updates)
        return ui_new, opt_state