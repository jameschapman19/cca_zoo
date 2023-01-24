import warnings
from typing import Union, Iterable

import numpy as np

from cca_zoo.models._search import _delta_search
from cca_zoo.utils import _process_parameter, _check_converged_weights
from ._pls_als import PLS_ALS


class SCCA_PMD(PLS_ALS):
    r"""
    Fits a Sparse CCA (Penalized Matrix Decomposition) model for 2 or more views.

    .. math::

        w_{opt}=\underset{w}{\mathrm{argmax}}\{ w_1^TX_1^TX_2w_2  \}\\

        \text{subject to:}

        w_i^Tw_i=1

        \|w_i\|<=c_i

    Parameters
    ----------
    latent_dims : int, default=1
        Number of latent dimensions to use in the model.
    scale : bool, default=True
        Whether to scale the data to unit variance.
    centre : bool, default=True
        Whether to centre the data to have zero mean.
    copy_data : bool, default=True
        Whether to copy the data or overwrite it.
    random_state : int, default=None
        Random seed for initialisation.
    deflation : str, default="cca"
        Deflation method to use. Options are "cca" and "pmd".
    tau : float or list of floats, default=None
        Regularisation parameter. If a single float is given, the same value is used for all views.
        If a list of floats is given, the values are used for each view respectively.
        If None, the value is set to 1.
    max_iter : int, default=100
        Maximum number of iterations to run.
    initialization : str or callable, default="pls"
        Method to use for initialisation. Options are "pls" and "random".
    tol : float, default=1e-9
        Tolerance for convergence.
    positive : bool or list of bools, default=False
        Whether to constrain the weights to be positive.
    verbose : int, default=0
        Verbosity level. 0 is silent, 1 prints progress.


    References
    ----------
    Witten, Daniela M., Robert Tibshirani, and Trevor Hastie. "A penalized matrix decomposition, with applications to sparse principal components and canonical correlation analysis." Biostatistics 10.3 (2009): 515-534.

    Examples
    --------
    >>> from cca_zoo.models import SCCA_PMD
    >>> import numpy as np
    >>> rng=np.random.RandomState(0)
    >>> X1 = rng.random((10,5))
    >>> X2 = rng.random((10,5))
    >>> model = SCCA_PMD(tau=[1,1],random_state=0)
    >>> model.fit((X1,X2)).score((X1,X2))
    array([0.81796873])
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
        positive: Union[Iterable[bool], bool] = None,
        verbose=0,
    ):
        self.tau = tau
        self.positive = positive
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
        if self.tau is None:
            warnings.warn(
                "tau parameter not set. Setting to tau=1 i.e. maximum regularisation of l1 norm"
            )
        self.tau = _process_parameter("tau", self.tau, 1, self.n_views)
        if any(tau < 0 or tau > 1 for tau in self.tau):
            raise ValueError(
                "All regularisation parameters should be between 0 and 1 "
                f"1. tau=[{self.tau}]"
            )
        self.positive = _process_parameter(
            "positive", self.positive, False, self.n_views
        )

    def _initialize(self, views):
        shape_sqrts = [np.sqrt(view.shape[1]) for view in views]
        self.t = [max(1, x * y) for x, y in zip(self.tau, shape_sqrts)]

    def _update(self, views, scores, weights):
        # Update each view using loop update function
        for view_index, view in enumerate(views):
            targets = np.ma.array(scores, mask=False)
            targets.mask[view_index] = True
            weights[view_index] = views[view_index].T @ targets.sum(axis=0).filled()
            weights[view_index] = _delta_search(
                weights[view_index],
                self.t[view_index],
                tol=self.tol,
            )
            _check_converged_weights(weights[view_index], view_index)
            scores[view_index] = views[view_index] @ weights[view_index]
        return scores, weights
