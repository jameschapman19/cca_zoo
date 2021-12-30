from abc import abstractmethod

import jax.numpy as jnp
import numpy as np
from cca_zoo.models import rCCA
from ccagame.baseexperiment import BaseExperiment
from jax import jit
from .utils import _TCC


class CCAExperiment(BaseExperiment):
    def __init__(
        self,
        mode,
        init_rng=None,
        num_devices=1,
        n_components=1,
        data=None,
        batch_size=0,
        path=None,
        TCC=False,
        **kwargs,
    ):

        super(CCAExperiment, self).__init__(
            mode=mode,
            init_rng=init_rng,
            num_devices=num_devices,
            n_components=n_components,
            data=data,
            batch_size=batch_size,
            cca=True,
            **kwargs,
        )
        """Constructs the experiment.
        Args:
          mode: A string, equivalent to FLAGS.jaxline_mode when running normally.
          init_rng: A `PRNGKey` to use for experiment initialization.
        """
        """Initialization function for a Jaxline experiment."""
        self.TCC = TCC

    def _init_ground_truth(self, X, Y):
        cca = rCCA(
            latent_dims=self.n_components, scale=False, centre=False, c=0.01
        ).fit((X, Y))
        self.correct_U, self.correct_V = cca.weights
        self.correct_U /= np.linalg.norm(self.correct_U, axis=0)
        self.correct_V /= np.linalg.norm(self.correct_V, axis=0)

    @abstractmethod
    def _update(self, views, global_step):
        raise NotImplementedError

    def _get_scalars(self):
        scalars = {}
        if self.TCC:
            scalars["TCC"] = _TCC(self.X_val, self.Y_val, self._U, self._V)
        scalars["correct x"] = self._correct_eigenvector_streak(self._U, self.correct_U)
        scalars["correct y"] = self._correct_eigenvector_streak(self._V, self.correct_V)
        scalars["subspace x"] = self._normalized_subspace_distance(
            self._U, self.correct_U
        )
        scalars["subspace y"] = self._normalized_subspace_distance(
            self._V, self.correct_V
        )
        return scalars

    def save_outputs(self):
        np.savetxt("U.csv", self._U, delimiter=",")
        np.savetxt("V.csv", self._V, delimiter=",")

    def evaluate(
        self,
        *,
        global_step: jnp.ndarray,
        rng: jnp.ndarray,
        writer,
    ):
        return {}
