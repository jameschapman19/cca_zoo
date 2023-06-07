from typing import Union

import numpy as np

from cca_zoo.models._plsmixin import PLSMixin
from cca_zoo.models._iterative._base import BaseDeflation, BaseLoop


class PLS_ALS(BaseDeflation, PLSMixin):
    r"""
    A class used to fit a PLS model by alternating least squares (ALS).

    This model finds the linear projections of two views that maximize their covariance while minimizing their residual variance.

    The objective function of PLS-ALS is:

    .. math::

        w_{opt}=\underset{w}{\mathrm{argmax}}\{ w_1^TX_1^TX_2w_2  \}\\

        \text{subject to:}

        w_i^Tw_i=1

    The algorithm alternates between updating :math:`w_1` and :math:`w_2` until convergence.

    Parameters
    ----------
    latent_dims : int, optional
        Number of latent dimensions to use, by default 1
    copy_data : bool, optional
        Whether to copy the data, by default True
    random_state : int, optional
        Random seed for reproducibility, by default None
    epochs : int, optional
        Number of iterations to run the algorithm, by default 100
    deflation : str, optional
        Deflation scheme to use, by default "cca"
    initialization : str, optional
        Initialization scheme to use, by default "pls"
    tol : float, optional
        Tolerance for convergence, by default 1e-3
    """

    def __init__(
        self,
        latent_dims: int = 1,
        copy_data=True,
        random_state=None,
        tol=1e-3,
        deflation="pls",
        accept_sparse=None,
        batch_size=None,
        dataloader_kwargs=None,
        epochs=100,
        val_split=None,
        learning_rate=1,
        initialization: Union[str, callable] = "random",
        callbacks=None,
        trainer_kwargs=None,
    ):
        super().__init__(
            latent_dims,
            copy_data,
            random_state,
            tol,
            deflation=deflation,
            accept_sparse=accept_sparse,
            batch_size=batch_size,
            dataloader_kwargs=dataloader_kwargs,
            epochs=epochs,
            val_split=val_split,
            learning_rate=learning_rate,
            initialization=initialization,
            callbacks=callbacks,
            trainer_kwargs=trainer_kwargs,
        )

    def _get_module(self, weights=None, k=None):
        return PlsAlsLoop(
            weights=weights,
            k=k,
        )

    def _more_tags(self):
        return {"multiview": True}


class PlsAlsLoop(BaseLoop):
    def training_step(self, batch, batch_idx):
        scores = np.stack(self(batch["views"]))
        # Update each view using loop update function
        for view_index, view in enumerate(batch["views"]):
            # create a mask that is True for elements not equal to k along dim k
            mask = np.arange(scores.shape[0]) != view_index
            # apply the mask to scores and sum along dim k
            target = np.sum(scores[mask], axis=0)
            self.weights[view_index] = np.cov(
                np.hstack((batch["views"][view_index], target[:, np.newaxis])).T
            )[:-1, -1]
            self.weights[view_index] /= np.linalg.norm(self.weights[view_index])
