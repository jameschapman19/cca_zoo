from typing import Union

import numpy as np

from cca_zoo.models._iterative._base import _BaseIterative


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
        tol: float = 1e-3,
        deflation="pls",
        verbose=0,
    ):
        super().__init__(
            latent_dims=latent_dims,
            scale=scale,
            centre=centre,
            copy_data=copy_data,
            deflation=deflation,
            max_iter=max_iter,
            initialization=initialization,
            tol=tol,
            random_state=random_state,
            verbose=verbose,
        )

    def _update(self, views, scores, weights):
        # Update each view using loop update function
        for view_index, view in enumerate(views):
            # mask off the current view and sum the rest
            targets = np.ma.array(scores, mask=False)
            targets.mask[view_index] = True
            weights[view_index] = views[view_index].T @ targets.sum(axis=0).filled()
            weights[view_index] /= np.linalg.norm(weights[view_index])
            scores[view_index] = views[view_index] @ np.squeeze(
                np.array(weights[view_index])
            )
        return scores, weights

    def _more_tags(self):
        return {"multiview": True}
