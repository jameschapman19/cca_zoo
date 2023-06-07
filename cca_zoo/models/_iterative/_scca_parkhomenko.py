from typing import Iterable, Union

import numpy as np
import pytorch_lightning as pl

from cca_zoo.models._iterative._base import BaseDeflation, BaseLoop
from cca_zoo.models._search import _softthreshold
from cca_zoo.utils import _process_parameter


class SCCA_Parkhomenko(BaseDeflation):
    r"""
    A class used to fit a sparse CCA (penalized CCA) model for two or more views.

    This model finds the linear projections of multiple views that maximize their pairwise correlations while enforcing sparsity constraints on the projection vectors.

    The objective function of sparse CCA is:

    .. math::

        w_{opt}=\underset{w}{\mathrm{argmax}}\{ w_1^TX_1^TX_2w_2 - \sum_i c_i\|w_i\|_1 \}\\

        \text{subject to:}

        w_i^Tw_i=1

    where :math:`c_i` are the sparsity parameters for each view.

    References
    ----------
    Parkhomenko, Elena, David Tritchler, and Joseph Beyene. "Sparse canonical correlation analysis with application to genomic data integration." Statistical applications in genetics and molecular biology 8.1 (2009).

    Examples
    --------
    >>> from cca_zoo.models import SCCA_Parkhomenko
    >>> import numpy as np
    >>> rng=np.random.RandomState(0)
    >>> X1 = rng.random((10,5))
    >>> X2 = rng.random((10,5))
    >>> model = SCCA_Parkhomenko(tau=[0.001,0.001],random_state=0)
    >>> model.fit((X1,X2)).score((X1,X2))
    array([0.81803527])
    """

    def __init__(
        self,
        latent_dims: int = 1,
        copy_data=True,
        random_state=None,
        deflation="cca",
        tau: Union[Iterable[float], float] = None,
        initialization: Union[str, callable] = "pls",
        tol: float = 1e-3,
        convergence_checking=False,
        patience=10,
        track=False,
        verbose=False,
    ):
        self.tau = tau
        super().__init__(
            latent_dims=latent_dims,
            copy_data=copy_data,
            initialization=initialization,
            tol=tol,
            random_state=random_state,
            deflation=deflation,
            convergence_checking=convergence_checking,
            patience=patience,
            track=track,
            verbose=verbose,
        )

    def _check_params(self):
        self.tau = _process_parameter("tau", self.tau, 0.0001, self.n_views_)
        if any(tau <= 0 for tau in self.tau):
            raise (
                "All regularisation parameters should be above 0. " f"tau=[{self.tau}]"
            )

    def _get_module(self, weights=None, k=None):
        return ParkhomenkoLoop(
            weights=weights,
            k=k,
            tau=self.tau,
            tol=self.tol,
        )

    def _more_tags(self):
        return {"multiview": True}


class ParkhomenkoLoop(BaseLoop):
    def __init__(self, weights, k=None, tau=None, tol=1e-3):
        super().__init__(weights=weights, k=k)
        self.tau = tau
        self.tol = tol

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
            self.weights[view_index] = _softthreshold(
                self.weights[view_index], self.tau[view_index] / 2
            )
            self.weights[view_index] /= np.linalg.norm(self.weights[view_index])
