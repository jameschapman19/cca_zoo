"""
Appgrad
"""
from functools import partial
from os import environ
import jax
from jax._src.lax.lax import sqrt
import jax.numpy as jnp
import optax

from ccagame.cca.utils import _get_AB
from . import CCAExperiment
import jax.scipy as jsp
from jax import jit


class AppGrad(CCAExperiment):
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
        c=None,
        **kwargs
    ):
        super(AppGrad, self).__init__(
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
        self._U = (
            jax.random.normal(self.local_rng, (self.n_components, self.dims[0])) / 10000
        )
        self._V = (
            jax.random.normal(self.local_rng, (self.n_components, self.dims[1])) / 10000
        )
        self._U_tilde=jnp.zeros_like(self._U)
        self._V_tilde=jnp.zeros_like(self._V)
        self._update_with_grads = jax.jit(
            jax.vmap(
                self._update_with_grads,
                in_axes=(0, 0, 0),
            )
        )
        self._optimizer = optax.sgd(learning_rate=learning_rate)
        self._opt_state_x = self._optimizer.init(self._U_tilde)
        self._opt_state_y = self._optimizer.init(self._V_tilde)
        self.c=c
        if self.c is None:
            self.c=[0,0]

    def _update(self, views, global_step):
        X_i, Y_i = views
        x_grads = self._grad(X_i, Y_i, self._V,self._U_tilde,self.c[0])
        self._U_tilde, self._opt_state_x = self._update_with_grads(
            self._U, x_grads, self._opt_state_x
        )
        self._U=self._normalize(X_i,self._U_tilde,self.c[0])
        y_grads = self._grad(Y_i, X_i, self._U,self._V_tilde,self.c[1])
        self._V_tilde, self._opt_state_y = self._update_with_grads(
            self._V, y_grads, self._opt_state_y
        )
        self._V=self._normalize(Y_i,self._V_tilde,self.c[1])


    @staticmethod
    @jit
    def _grad(X_i,Y_i,V,U_tilde,c):
        n=X_i.shape[0]
        grads=(U_tilde@X_i.T@X_i-V@Y_i.T@X_i)/n+U_tilde*c
        return grads

    @partial(jit, static_argnums=(0))
    def _update_with_grads(self, wi, grads, opt_state):
        # we have gradient of utilities so we negate for gradient descent
        updates, opt_state = self._optimizer.update(-grads, opt_state)
        wi_new = optax.apply_updates(wi, updates)
        return wi_new, opt_state

    @staticmethod
    @jit
    def _normalize(X_i,U,c):
        n=X_i.shape[0]
        M=U@X_i.T@X_i@U.T+n*c*jnp.eye(U.shape[0])
        return _sqrtm(M).T@U

@jit
def _sqrtm(M):
    U,S,_=jnp.linalg.svd(M)
    return U@jnp.diag(1/jnp.sqrt(S))