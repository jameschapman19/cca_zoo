"""
SAA
"""
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import optax
from cca_zoo.models import MCCA
from functools import partial
from jax import jit
from re import I
from scipy.linalg import sqrtm

from ._ccamixin import _CCAMixin
from .._baseexperiment import _BaseExperiment
from .._utils import _get_AB
from .._utils import invsqrtm
from ..datasets._utils import data_stream


class SAA(_CCAMixin, _BaseExperiment):
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

    def _update(self, views, global_step):
        if global_step == 0 or (global_step + 1) % self.config.val_interval == 0:
            self._U, self._V = (
                MCCA(latent_dims=self.config.n_components)
                .fit(
                    (
                        self.X[: (1 + global_step[0]) * self.config.batch_size],
                        self.Y[: (1 + global_step[0]) * self.config.batch_size],
                    )
                )
                .weights
            )
            self._U = self._U.T
            self._V = self._V.T
