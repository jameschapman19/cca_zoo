from abc import abstractmethod

import jax.numpy as jnp
import numpy as np
from ccagame.baseexperiment import BaseExperiment
from jax import jit
from cca_zoo.models import PLS


class PLSExperiment(BaseExperiment):
    def __init__(
        self,
        mode,
        init_rng=None,
        num_devices=1,
        n_components=1,
        data=None,
        batch_size=0,
        path=None,
        num_batches=None,
        TV=False,
        **kwargs,
    ):
        super(PLSExperiment, self).__init__(
            mode=mode,
            init_rng=init_rng,
            num_devices=num_devices,
            n_components=n_components,
            data=data,
            batch_size=batch_size,
            **kwargs,
        )
        """Constructs the experiment.
        Args:
          mode: A string, equivalent to FLAGS.jaxline_mode when running normally.
          init_rng: A `PRNGKey` to use for experiment initialization.
        """
        """Initialization function for a Jaxline experiment."""
        self.dims = [self.X.shape[1], self.Y.shape[1]]
        self.TV = TV
        if self.validate:
            self._init_ground_truth(self.X, self.Y)

    def _init_ground_truth(self, X, Y):
        pls = PLS(
            latent_dims=self.n_components, scale=False, centre=False
        ).fit((X, Y))
        self.correct_U, self.correct_V = pls.weights
        self.correct_U/=np.linalg.norm(self.correct_U,axis=0)
        self.correct_V/=np.linalg.norm(self.correct_V,axis=0)
        if self.TV:
            self.TV_train = self._TV(self.correct_U.T,self.correct_V.T,self.X,self.Y)
            self.TV_val = self._TV(self.correct_U.T,self.correct_V.T,self.X_val,self.Y_val)

    @abstractmethod
    def _update(self, views, global_step):
        raise NotImplementedError

    def _get_scalars(self):#(self.X_val@self.correct_U).T@(self.X_val@self.correct_U)/4000
        scalars = {}#(self.X_val@self._U.T).T@(self.X_val@self._U.T)/4000
        if self.TV:#self._TV(self.correct_U.T, self.correct_V.T, self.X, self.Y)
            scalars["tv train"] = self._TV(self._U, self._V, self.X, self.Y)
            scalars["tv val"] = self._TV(self._U, self._V, self.X_val, self.Y_val)
        scalars["correct_x"] = self._correct_eigenvector_streak(self._U, self.correct_U)
        scalars["correct_y"] = self._correct_eigenvector_streak(self._V, self.correct_V)
        scalars["subspace_x"] = self._normalized_subspace_distance(
            self._U, self.correct_U
        )
        scalars["subspace_y"] = self._normalized_subspace_distance(
            self._V, self.correct_V
        )
        return scalars

    @staticmethod
    #@jit
    def _TV(U, V, X_val, Y_val):
        dof = X_val.shape[0]
        Zx = X_val @ U.T
        Zy = Y_val @ V.T
        return jnp.sum(jnp.abs(jnp.linalg.svd(Zx.T @ Zy)[1])) / dof

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
