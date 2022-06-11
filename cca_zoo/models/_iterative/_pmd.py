import warnings
from typing import Union, Iterable

import numpy as np

from cca_zoo.models._iterative.utils import _delta_search
from cca_zoo.utils import _process_parameter, _check_converged_weights
from ._base import _BaseIterative
from ._pls_als import _PLSInnerLoop


class SCCA_PMD(_BaseIterative):
    r"""
    Fits a Sparse CCA (Penalized Matrix Decomposition) model.

    :Maths:

    .. math::

        w_{opt}=\underset{w}{\mathrm{argmax}}\{ w_1^TX_1^TX_2w_2  \}\\

        \text{subject to:}

        w_i^Tw_i=1

        \|w_i\|<=c_i

    :Citation:

    Witten, Daniela M., Robert Tibshirani, and Trevor Hastie. "A penalized matrix decomposition, with applications to sparse principal components and canonical correlation analysis." Biostatistics 10.3 (2009): 515-534.

    :Example:

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
        """
        Constructor for SCCA_PMD

        :param latent_dims: number of latent dimensions to fit
        :param scale: normalize variance in each column before fitting
        :param centre: demean data by column before fitting (and before transforming out of sample
        :param copy_data: If True, views will be copied; else, it may be overwritten
        :param random_state: Pass for reproducible output across multiple function calls
        :param c: l1 regularisation parameter between 1 and sqrt(number of features) for each view
        :param max_iter: the maximum number of iterations to perform in the inner optimization loop
        :param initialization: either string from "pls", "cca", "random", "uniform" or callable to initialize the score variables for _iterative methods
        :param tol: tolerance value used for early stopping
        :param positive: constrain model weights to be positive
        """
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
        """
        :param view_index: index of view being updated
        :return: updated weights
        """
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
