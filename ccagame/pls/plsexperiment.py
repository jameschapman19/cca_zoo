from abc import abstractmethod

import jax.numpy as jnp
import numpy as np
from cca_zoo.models import PLS
from ccagame.baseexperiment import BaseExperiment
from jax import jit
from ccagame.datasets.xrmb import xrmb_true

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
        if self.val_interval > 0:
            self._init_ground_truth(self.X, self.Y)

    def _init_ground_truth(self, X, Y):
        if self.data=='xrmb':
            self.correct_U,self.correct_V=xrmb_true()
        else:
            U, _, Vt = jnp.linalg.svd(X.T @ Y)
            self.correct_U = U[
                :, : self.n_components
            ]
            self.correct_V = Vt[: self.n_components, :].T
        if self.TV:
            if self.data!='xrmb':
                self.TV_train = self._TV(self.correct_U.T, self.correct_V.T, self.X, self.Y)
            self.TV_val = self._TV(
                self.correct_U.T, self.correct_V.T, self.X_val, self.Y_val
            )

    @abstractmethod
    def _update(self, views, global_step):
        raise NotImplementedError

    def _get_scalars(self, global_step):
        scalars = {}
        if global_step == 0 or (global_step + 1) % self.val_interval == 0:
            if self.TV:
                if self.data!='xrmb':
                    scalars["TV train"] = self._TV(self._U, self._V, self.X, self.Y)
                    scalars["PV train"] = scalars["TV train"] / self.TV_train
                scalars["TV val"] = self._TV(self._U, self._V, self.X_val, self.Y_val)
                scalars["PV val"] = scalars["TV val"] / self.TV_val
            scalars["correct x"] = self._correct_eigenvector_streak(
                self._U, self.correct_U
            )
            scalars["correct y"] = self._correct_eigenvector_streak(
                self._V, self.correct_V
            )
            scalars["sum cosine similarities x"] = self._sum_cosine_similarities(
                self._U, self.correct_U
            )
            scalars["sum cosine similarities y"] = self._sum_cosine_similarities(
                self._V, self.correct_V
            )
            scalars["subspace x"] = self._normalized_subspace_distance(
                self._U, self.correct_U
            )
            scalars["subspace y"] = self._normalized_subspace_distance(
                self._V, self.correct_V
            )
        return scalars

    @staticmethod
    @jit
    def _TV(U, V, X_val, Y_val):
        dof = X_val.shape[0]
        Qu, Ru = jnp.linalg.qr(U.T)
        Su = jnp.sign(jnp.sign(jnp.diag(Ru)) + 0.5)
        Qv, Rv = jnp.linalg.qr(V.T)
        Sv = jnp.sign(jnp.sign(jnp.diag(Rv)) + 0.5)
        return jnp.trace(jnp.atleast_2d((Qu @ jnp.diag(Su)).T @ X_val.T @ Y_val @ (Qv @ jnp.diag(Sv)))) / dof

    def save_outputs(self):
        np.savetxt("U.csv", self._U, delimiter=",")
        np.savetxt("V.csv", self._V, delimiter=",")

    @staticmethod
    @jit
    def _sum_cosine_similarities(U, U_correct):
        n_components = U.shape[0]
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
