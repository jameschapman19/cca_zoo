from os import environ
import jax
import jax.numpy as jnp
import optax
from . import CCAExperiment
import jax.scipy as jsp

class GenOja(CCAExperiment):
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
        self._W = jax.random.normal(self.local_rng, (self.n_components, self.dims[0]+self.dims[1]))
        self._optimizer = optax.sgd(
            learning_rate=learning_rate, momentum=momentum, nesterov=nesterov
        )
        self._opt_state = self._optimizer.init(self._W)


    # @partial(jit, static_argnums=(0))
    def _update(self, views, global_step):
        X_i, Y_i = views
        X_i = jnp.reshape(X_i, (self.num_devices, -1, self.dims[0]))
        Y_i = jnp.reshape(Y_i, (self.num_devices, -1, self.dims[1]))
        B=jsp.linalg.block_diag(X_i.T@X_i,Y_i.T@Y_i)
        A=jnp.hstack((X_i,Y_i)).T@jnp.hstack((X_i,Y_i))
        A=A-B
        Y=self._W@A@self._W.T
        w_grad=B@self._W@Y-A@self._W.T
        updates_w, self._opt_state = self._optimizer.update(
            w_grad, self._opt_state
        )
        self._W = optax.apply_updates(self._W, updates_w)
        self._W = jnp.linalg.qr(self._W.T)[0].T
        self._U=self._V[:,:self.dims[0]]
        self._V=self._V[:,:self.dims[1]]
