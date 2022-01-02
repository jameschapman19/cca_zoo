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
        self.dims = [self.X.shape[1], self.Y.shape[1]]
        self.TCC = TCC
        if self.validate:
            self._init_ground_truth(self.X, self.Y)
            
    def _init_ground_truth(self, X, Y):
        cca = rCCA(
            latent_dims=self.n_components, scale=False, centre=False, c=0
        ).fit((X, Y))
        self.correct_U, self.correct_V = cca.weights
        self.correct_Zx,self.correct_Zy = cca.transform((self.X_val,self.Y_val))
        if self.TCC:
            self.TCC_train = _TCC(self.X,self.Y,self.correct_U.T,self.correct_V.T)
            self.TCC_val = _TCC(self.X_val,self.Y_val,self.correct_U.T,self.correct_V.T)

    @abstractmethod
    def _update(self, views, global_step):
        raise NotImplementedError

    def _get_scalars(self):#(self.X_val@self.correct_U).T@(self.X_val@self.correct_U)
        scalars = {}#np.corrcoef(self.X_val@self.correct_U, self.Y_val@self.correct_V)[16:,:16]
        if self.TCC:#_TCC(self.X_val, self.Y_val, self.correct_U.T, self.correct_V.T)
            scalars["TCC train"] = _TCC(self.X, self.Y, self._U, self._V)
            scalars["TCC val"] = _TCC(self.X_val, self.Y_val, self._U, self._V)
        scalars["correct x"] = self._correct_eigenvector_streak(self._U, self.correct_U)
        scalars["correct y"] = self._correct_eigenvector_streak(self._V, self.correct_V)
        scalars["sum cosine similarities x"] = self._sum_cosine_similarities(self._U, self.correct_U)
        scalars["sum cosine similarities y"] = self._sum_cosine_similarities(self._V, self.correct_V)
        return scalars

    @staticmethod
    #@jit
    def _sum_cosine_similarities(U, U_correct):
        n_components = U.shape[0]
        cosine_similarities = jnp.diag(
            jnp.corrcoef(U.T, U_correct, rowvar=False)[n_components:, :n_components]
        )
        return jnp.sum(jnp.abs(cosine_similarities))

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
