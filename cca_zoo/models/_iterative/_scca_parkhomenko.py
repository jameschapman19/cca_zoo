from typing import Union, Iterable

import numpy as np

from cca_zoo.models._iterative._base import _BaseIterative
from cca_zoo.models._search import _softthreshold
from cca_zoo.utils import _process_parameter, _check_converged_weights


class SCCA_Parkhomenko(_BaseIterative):
    r"""
    Fits a sparse CCA (penalized CCA) model for 2 or more views.

    .. math::

        w_{opt}=\underset{w}{\mathrm{argmax}}\{ w_1^TX_1^TX_2w_2  \} + c_i\|w_i\|\\

        \text{subject to:}

        w_i^Tw_i=1

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
        scale: bool = True,
        centre=True,
        copy_data=True,
        random_state=None,
        deflation="cca",
        tau: Union[Iterable[float], float] = None,
        max_iter: int = 100,
        initialization: Union[str, callable] = "pls",
        tol: float = 1e-3,
        verbose=0,
    ):
        self.tau = tau
        super().__init__(
            latent_dims=latent_dims,
            scale=scale,
            centre=centre,
            copy_data=copy_data,
            max_iter=max_iter,
            initialization=initialization,
            tol=tol,
            random_state=random_state,
            deflation=deflation,
            verbose=verbose,
        )

    def _check_params(self):
        self.tau = _process_parameter("tau", self.tau, 0.0001, self.n_views)
        if any(tau <= 0 for tau in self.tau):
            raise (
                "All regularisation parameters should be above 0. " f"tau=[{self.tau}]"
            )

    def _update(self, views, scores, weights):
        for view_index, view in enumerate(views):
            # mask off the current view and sum the rest
            targets = np.ma.array(scores, mask=False)
            targets.mask[view_index] = True
            w = views[view_index].T @ targets.sum(axis=0).filled()
            w /= np.linalg.norm(w)
            _check_converged_weights(w, view_index)
            w = _softthreshold(w, self.tau[view_index] / 2)
            weights[view_index] = w / np.linalg.norm(w)
            _check_converged_weights(w, view_index)
            scores[view_index] = views[view_index] @ weights[view_index]
        return scores, weights

    def _more_tags(self):
        return {"multiview": True}
