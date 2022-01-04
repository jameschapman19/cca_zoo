from abc import abstractmethod

import jax.numpy as jnp
import numpy as np
from ccagame.baseexperiment import BaseExperiment
from jax import jit
from cca_zoo.models import PLS


class PLSExperiment(BaseExperiment):
    NON_BROADCAST_CHECKPOINT_ATTRS = {"_U": "U", "_V": "V"}
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
        if self.val_interval>0:
            self._init_ground_truth(self.X, self.Y)

    def _init_ground_truth(self, X, Y):
        U,_,Vt=jnp.linalg.svd(X.T@Y)
        self.correct_U=U[:,:self.n_components]#jnp.linalg.norm(self.correct_U,axis=0)
        self.correct_V=Vt[:self.n_components,:].T
        if self.TV:
            self.TV_train = self._TV(self.correct_U.T,self.correct_V.T,self.X,self.Y)
            self.TV_val = self._TV(self.correct_U.T,self.correct_V.T,self.X_val,self.Y_val)

    @abstractmethod
    def _update(self, views, global_step):
        raise NotImplementedError

    def _get_scalars(self,global_step):
        scalars = {}
        if (global_step+1)%self.val_interval==0:
            if self.TV:
                scalars["tv train"] = self._TV(self._U, self._V, self.X, self.Y)
                scalars["tv val"] = self._TV(self._U, self._V, self.X_val, self.Y_val)
            scalars["correct x"] = self._correct_eigenvector_streak(self._U, self.correct_U)
            scalars["correct y"] = self._correct_eigenvector_streak(self._V, self.correct_V)
            scalars["sum cosine similarities x"] = self._sum_cosine_similarities(self._U, self.correct_U)
            scalars["sum cosine similarities y"] = self._sum_cosine_similarities(self._V, self.correct_V)
        return scalars

    @staticmethod
    @jit
    def _TV(U, V, X_val, Y_val):
        dof = X_val.shape[0]
        Zx = X_val @ U.T
        Zy = Y_val @ V.T
        return jnp.sum(jnp.abs(jnp.linalg.svd(Zx.T @ Zy)[1])) / dof

    def save_outputs(self):
        np.savetxt("U.csv", self._U, delimiter=",")
        np.savetxt("V.csv", self._V, delimiter=",")

    @staticmethod
    #@jit
    def _sum_cosine_similarities(U, U_correct):#U@U.T
        n_components = U.shape[0]#U_correct.T@U_correct
        cosine_similarities = jnp.diag(
            jnp.corrcoef(U.T, U_correct, rowvar=False)[n_components:, :n_components]
        )
        return jnp.sum(jnp.abs(cosine_similarities))

    def evaluate(
        self,
        *,
        global_step: jnp.ndarray,
        rng: jnp.ndarray,
        writer,
    ):
        return {}
