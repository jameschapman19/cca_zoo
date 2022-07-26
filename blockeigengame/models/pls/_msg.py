import jax.numpy as jnp

from ._incremental import Incremental
from ..._utils import _capping

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
        self._U, self._V, self._S = self._grads(
            jnp.sqrt(self.config.learning_rate) * X_i, jnp.sqrt(self.config.learning_rate) * Y_i, self._U, self._V,
            self.config.batch_size * (global_step) * self._S
        )
        self._S = _capping(self._S / (self.config.batch_size * (global_step + 1)), self.config.n_components)
