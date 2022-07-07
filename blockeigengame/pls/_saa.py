import jax
import jax.numpy as jnp

from ._plsmixin import _PLSMixin
from .._baseexperiment import _BaseExperiment


class SAA(_PLSMixin, _BaseExperiment):
    def __init__(self, mode, init_rng, config):
        super(SAA, self).__init__(mode, init_rng, config)
        """Constructs the experiment.
        Args:
          mode: A string, equivalent to FLAGS.jaxline_mode when running normally.
          init_rng: A `PRNGKey` to use for experiment initialization.
        """
        """Initialization function for a Jaxline experiment."""

    def _init_train(self):
        self._init_ground_truth()
        views = next(self._train_input)
        self._U = jax.random.normal(
            self.init_rng, (self.config.n_components, views[0].shape[1])
        )
        self._U /= jnp.linalg.norm(self._U, axis=1, keepdims=True)
        self._V = jax.random.normal(
            self.init_rng, (self.config.n_components, views[1].shape[1])
        )
        self._V /= jnp.linalg.norm(self._V, axis=1, keepdims=True)
        self.Cxy = jnp.zeros((views[0].shape[1], views[1].shape[1]))

    def _update(self, views, global_step):
        X_i, Y_i = views
        self.Cxy = (
            ((global_step + 1) - 1) * self.Cxy + (X_i.T @ Y_i / X_i.shape[0])
        ) / (global_step + 1)
        U, _, Vt = jnp.linalg.svd(self.Cxy)
        self._U = U[:, : self.config.n_components].T
        self._V = Vt[: self.config.n_components]
