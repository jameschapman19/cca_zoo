from abc import abstractmethod
from typing import Union

import numpy as np

from ._base import _BaseInnerLoop, _BaseIterative


class PLS_ALS(_BaseIterative):
    r"""
    A class used to fit a PLS model to two or more views of data.

    Fits a partial least squares model with CCA deflation by NIPALS algorithm

    .. math::

        w_{opt}=\underset{w}{\mathrm{argmax}}\{\sum_i\sum_{j\neq i} \|X_iw_i-X_jw_j\|^2\}\\

        \text{subject to:}

        w_i^Tw_i=1

    Can also be used with more than two views

    Parameters
    ----------
    latent_dims : int, optional
        Number of latent dimensions to use, by default 1
    scale : bool, optional
        Whether to scale the data, by default True
    centre : bool, optional
        Whether to centre the data, by default True
    copy_data : bool, optional
        Whether to copy the data, by default True
    random_state : int, optional
        Random state for reproducibility, by default None
    max_iter : int, optional
        Maximum number of iterations, by default 100
    initialization : Union[str, callable], optional
        Initialization method, by default "random"
    tol : float, optional
        Tolerance for convergence, by default 1e-9
    verbose : int, optional
        Verbosity level, by default 0

    Examples
    --------

    >>> from cca_zoo.models import PLS
    >>> import numpy as np
    >>> rng=np.random.RandomState(0)
    >>> X1 = rng.random((10,5))
    >>> X2 = rng.random((10,5))
    >>> model = PLS_ALS(random_state=0)
    >>> model.fit((X1,X2)).score((X1,X2))
    array([0.81796854])
    """

    def __init__(
        self,
        latent_dims: int = 1,
        scale: bool = True,
        centre=True,
        copy_data=True,
        random_state=None,
        max_iter: int = 100,
        initialization: Union[str, callable] = "random",
        tol: float = 1e-9,
        verbose=0,
    ):
        super().__init__(
            latent_dims=latent_dims,
            scale=scale,
            centre=centre,
            copy_data=copy_data,
            deflation="pls",
            max_iter=max_iter,
            initialization=initialization,
            tol=tol,
            random_state=random_state,
            verbose=verbose,
        )

    def _set_loop_params(self):
        self.loop = _PLSInnerLoop(
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            verbose=self.verbose,
        )


class _PLSInnerLoop(_BaseInnerLoop):
    def __init__(
        self,
        max_iter: int = 100,
        tol=1e-9,
        random_state=None,
        verbose=0,
    ):
        super().__init__(
            max_iter=max_iter, tol=tol, random_state=random_state, verbose=verbose
        )

    def _inner_iteration(self, views):
        # Update each view using loop update function
        for i, view in enumerate(views):
            # if no nans
            if np.isnan(self.scores).sum() == 0:
                self._update_view(views, i)

    @abstractmethod
    def _update_view(self, views, view_index: int):
        # mask off the current view and sum the rest
        targets = np.ma.array(self.scores, mask=False)
        targets.mask[view_index] = True
        self.weights[view_index] = views[view_index].T @ targets.sum(axis=0).filled()
        self.weights[view_index] /= np.linalg.norm(self.weights[view_index])
        self.scores[view_index] = views[view_index] @ np.squeeze(
            np.array(self.weights[view_index])
        )
