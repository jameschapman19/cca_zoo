from os import stat
import jax
import jax.numpy as jnp

from ccagame.pls.utils import incrsvd
from . import PLSExperiment
from jax import jit


class Incremental(PLSExperiment):
    NON_BROADCAST_CHECKPOINT_ATTRS = {"_U": "U", "_V": "V"}

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
        self._U = jax.random.normal(self.init_rng, (self.n_components, self.dims[0]))
        self._U /= jnp.linalg.norm(self._U, axis=1, keepdims=True)
        self._V = jax.random.normal(self.init_rng, (self.n_components, self.dims[1]))
        self._V /= jnp.linalg.norm(self._V, axis=1, keepdims=True)
        self._S = jnp.zeros(self.n_components)
        if (max(self.dims[0],self.dims[1])*min(self.dims[0],self.dims[1])**2)<((self.n_components+batch_size)**3):
            self._grads=self._mat_grads
        else:
            self._grads=self._incr_grads

    def _update(self, views, global_step):
        X_i, Y_i = views
        self._U, self._V, self._S = self._grads(X_i, Y_i, self._U, self._V, self.batch_size*(global_step+1)*self._S)
        self._S/=(self.batch_size*(global_step+2))

    @staticmethod
    @jit
    def _incr_grads(X_i, Y_i, U, V, S):
        x_hat = X_i @ U.T
        x_orth = X_i - x_hat @ U
        y_hat = Y_i @ V.T
        y_orth = Y_i - y_hat @ V
        return incrsvd(x_hat,y_hat,x_orth,y_orth,U,V,S)

    @staticmethod
    @jit
    def _mat_grads(X_i, Y_i, U, V, S):
        n_components=U.shape[0]
        M_hat=X_i.T@Y_i+U.T@jnp.diag(S)@V
        U,S,Vt=jnp.linalg.svd(M_hat)
        return U[:, :n_components].T,Vt[:n_components,:],S[:n_components]