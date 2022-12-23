import warnings
from typing import Union, Iterable

import numpy as np

from cca_zoo.utils import _process_parameter, _check_converged_weights
from ._base import _BaseIterative
from ._pls_als import _PLSInnerLoop
from .._search import _delta_search


class SCCA_PMD(_BaseIterative):
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
    c : float or list of floats, default=None
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
    >>> model = SCCA_PMD(c=[1,1],random_state=0)
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
        c: Union[Iterable[float], float] = None,
        max_iter: int = 100,
        initialization: Union[str, callable] = "pls",
        tol: float = 1e-9,
        positive: Union[Iterable[bool], bool] = None,
        verbose=0,
    ):
        self.c = c
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

    def _set_loop_params(self):
        self.loop = _PMDInnerLoop(
            max_iter=self.max_iter,
            c=self.c,
            tol=self.tol,
            positive=self.positive,
            random_state=self.random_state,
            verbose=self.verbose,
        )

    def _check_params(self):
        if self.c is None:
            warnings.warn(
                "c parameter not set. Setting to c=1 i.e. maximum regularisation of l1 norm"
            )
        self.c = _process_parameter("c", self.c, 1, self.n_views)
        if any(c < 0 or c > 1 for c in self.c):
            raise ValueError(
                "All regularisation parameters should be between 0 and 1 "
                f"1. c=[{self.c}]"
            )
        self.positive = _process_parameter(
            "positive", self.positive, False, self.n_views
        )


class _PMDInnerLoop(_PLSInnerLoop):
    def __init__(
        self,
        max_iter: int = 100,
        tol=1e-9,
        c=None,
        positive=None,
        random_state=None,
        verbose=0,
    ):
        super().__init__(
            max_iter=max_iter, tol=tol, random_state=random_state, verbose=verbose
        )
        self.c = c
        self.positive = positive

    def _initialize(self, views):
        shape_sqrts = [np.sqrt(view.shape[1]) for view in views]
        self.t = [max(1, x * y) for x, y in zip(self.c, shape_sqrts)]

    def _update_view(self, views, view_index: int):
        # mask off the current view and sum the rest
        targets = np.ma.array(self.scores, mask=False)
        targets.mask[view_index] = True
        self.weights[view_index] = views[view_index].T @ targets.sum(axis=0).filled()
        self.weights[view_index] = _delta_search(
            self.weights[view_index],
            self.t[view_index],
            positive=self.positive[view_index],
            tol=self.tol,
        )
        _check_converged_weights(self.weights[view_index], view_index)
        self.scores[view_index] = views[view_index] @ self.weights[view_index]
