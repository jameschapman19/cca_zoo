"""
MSG
"""
from jax import jit

from ._incremental import Incremental
from .._utils import incrsvd, _capping


class MSG(Incremental):
    def __init__(self, mode, init_rng, config):
        super(MSG, self).__init__(mode, init_rng, config)
        """Constructs the experiment.
        Args:
          mode: A string, equivalent to FLAGS.jaxline_mode when running normally.
          init_rng: A `PRNGKey` to use for experiment initialization.
        """
        """Initialization function for a Jaxline experiment."""

    def _update(self, views, global_step):
        X_i, Y_i = views
        self.Uxx, self.Sxx = self.update_cov(self.Uxx, self.config.batch_size * (global_step) * self.Sxx, X_i)
        self.Uyy, self.Syy = self.update_cov(self.Uyy, self.config.batch_size * (global_step) * self.Syy, Y_i)
        self.Sxx /= (self.config.batch_size * (global_step + 1))
        self.Syy /= (self.config.batch_size * (global_step + 1))
        Wx = self._get_w(X_i, self.Uxx, self.Sxx)
        Wy = self._get_w(Y_i, self.Uyy, self.Syy)
        self._U, self._V, self._S = self._grads(self._U, self._V, self.config.batch_size * (global_step) * self._S, Wx,
                                                Wy)
        self._S = _capping(self._S / (self.config.batch_size * (global_step + 1)), self.config.n_components)

    @staticmethod
    @jit
    def update_cov(U, S, X):
        x_projected = X @ U.T
        x_leftover = X - x_projected @ U
        U, _, S = incrsvd(x_projected, x_projected, x_leftover, x_leftover, U, U, S)
        return U, S
