import jax
import jax.numpy as jnp
from . import PLSExperiment
from functools import partial
from jax import jit

class Incremental(PLSExperiment):
    def __init__(
        self,
        mode,
        init_rng=None,
        num_devices=1,
        n_components=1,
        data=None,
        batch_size=1,
        **kwargs
    ):
        super(Incremental, self).__init__(
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
        self._U=(1/jnp.linalg.norm(self._U,axis=1)*self._U.T).T
        self._V = jax.random.normal(self.local_rng, (self.n_components, self.dims[1]))
        self._V=(1/jnp.linalg.norm(self._V,axis=1)*self._V.T).T
        self._S = jax.random.normal(self.local_rng, (self.n_components,))

    #@partial(jit, static_argnums=(0))
    def _update(self, views, global_step):
        X_i, Y_i = views
        xhat = X_i @ self._U.T
        x_orth = X_i - X_i @ self._U.T @ self._U
        yhat = Y_i @ self._V.T
        y_orth = Y_i - Y_i @ self._V.T @ self._V
        Q = jnp.vstack(
            (
                jnp.hstack(
                    (
                        jnp.diag(self._S) + xhat.T @ yhat,
                        jnp.linalg.norm(y_orth) * xhat.T,
                    )
                ),
                jnp.hstack(
                    (
                        jnp.linalg.norm(x_orth) * yhat,
                        jnp.atleast_2d(
                            jnp.linalg.norm(x_orth) * jnp.linalg.norm(y_orth)
                        ),
                    )
                ),
            )
        )
        U_, S, Vt_ = jnp.linalg.svd(Q)
        self._U = U_[:, : self.n_components].T @ jnp.vstack((self._U, x_orth / jnp.linalg.norm(x_orth)))
        self._V = Vt_.T[:, : self.n_components].T @ jnp.vstack((self._V, y_orth / jnp.linalg.norm(y_orth)))
        self._S = S[: self.n_components]
