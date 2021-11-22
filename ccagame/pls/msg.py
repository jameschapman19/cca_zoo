from os import environ
import jax
import jax.numpy as jnp
import optax
from . import PLSExperiment
from functools import partial
from jax import jit



class MSG(PLSExperiment):
    def __init__(
        self,
        mode,
        init_rng=None,
        num_devices=1,
        n_components=1,
        dims=None,
        data=None,
        learning_rate=1e-3,
        momentum=0.9,
        nesterov=True,
        batch_size=0,
        **kwargs
    ):
        super(MSG, self).__init__(
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
        self._U = jax.random.normal(self.local_rng, (self.n_components, dims[0]))
        self._V = jax.random.normal(self.local_rng, (self.n_components, dims[1]))
        self._M = self._U.T@self._V
        self._optimizer =  optax.sgd(learning_rate=learning_rate, momentum=momentum, nesterov=nesterov)
        self._opt_state = self._optimizer.init(self._M)

    @partial(jit, static_argnums=(0))
    def _update(self, views, global_step):
        X_i, Y_i = views
        self._M=self._U.T@self._V
        grads=X_i.T@Y_i
        updates, self._opt_state = self._optimizer.update(grads, self._opt_state)
        self._M = optax.apply_updates(self._M, updates)
        U,_,Vt=jnp.linalg.svd(self._M)
        self._U=U[:,:self.n_components].T
        self._V=Vt[:self.n_components]
