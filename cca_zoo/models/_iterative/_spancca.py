from typing import Union, Iterable

from . import _BaseIterative
from . import _BaseInnerLoop
from cca_zoo.models._iterative.utils import _support_soft_thresh, _delta_search
from cca_zoo.utils import _process_parameter
import numpy as np


class SpanCCA(_BaseIterative):
    r"""
    Fits a Sparse CCA model using SpanCCA.

    .. math::

        w_{opt}=\underset{w}{\mathrm{argmax}}\{\sum_i\sum_{j\neq i} \|X_iw_i-X_jw_j\|^2 + \text{l1_ratio}\|w_i\|_1\}\\

        \text{subject to:}

        w_i^TX_i^TX_iw_i=1

    :Citation:

    Asteris, Megasthenis, et al. "A simple and provable algorithm for sparse diagonal CCA." International Conference on Machine Learning. PMLR, 2016.


    :Example:

    >>> from cca_zoo.models import SpanCCA
    >>> import numpy as np
    >>> rng=np.random.RandomState(0)
    >>> X1 = rng.random((10,5))
    >>> X2 = rng.random((10,5))
    >>> model = SpanCCA(regularisation="l0", c=[2, 2])
    >>> model.fit((X1,X2)).score((X1,X2))
    array([0.84556666])
    """

    def __init__(
        self,
        latent_dims: int = 1,
        scale: bool = True,
        centre=True,
        copy_data=True,
        max_iter: int = 100,
        initialization: str = "uniform",
        tol: float = 1e-9,
        regularisation="l0",
        c: Union[Iterable[Union[float, int]], Union[float, int]] = None,
        rank=1,
        positive: Union[Iterable[bool], bool] = None,
        random_state=None,
        deflation="cca",
    ):
        """

        :param latent_dims: number of latent dimensions to fit
        :param scale: normalize variance in each column before fitting
        :param centre: demean data by column before fitting (and before transforming out of sample
        :param copy_data: If True, X will be copied; else, it may be overwritten
        :param random_state: Pass for reproducible output across multiple function calls
        :param max_iter: the maximum number of iterations to perform in the inner optimization loop
        :param initialization: either string from "pls", "cca", "random", "uniform" or callable to initialize the score variables for _iterative methods
        :param tol: tolerance value used for early stopping
        :param regularisation:
        :param c: regularisation parameter
        :param rank: rank of the approximation
        :param positive: constrain weights to be positive
        """
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
        )
        self.c = c
        self.regularisation = regularisation
        self.rank = rank
        self.positive = positive

    def _set_loop_params(self):
        self.loop = _SpanCCAInnerLoop(
            max_iter=self.max_iter,
            c=self.c,
            tol=self.tol,
            regularisation=self.regularisation,
            rank=self.rank,
            random_state=self.random_state,
            positive=self.positive,
        )


class _SpanCCAInnerLoop(_BaseInnerLoop):
    def __init__(
        self,
        max_iter: int = 100,
        tol=1e-9,
        c=None,
        regularisation="l0",
        rank=1,
        random_state=None,
        positive=False,
    ):
        super().__init__(
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )
        self.c = c
        self.regularisation = regularisation
        self.rank = rank
        self.positive = positive

    def _check_params(self):
        """check number of views=2"""
        if len(self.views) != 2:
            raise ValueError(f"SpanCCA requires only 2 views")
        cov = self.views[0].T @ self.views[1] / self.n
        # Perform SVD on im and obtain individual matrices
        P, D, Q = np.linalg.svd(cov, full_matrices=True)
        self.P = P[:, : self.rank]
        self.D = D[: self.rank]
        self.Q = Q[: self.rank, :].T
        self.max_obj = 0
        if self.regularisation == "l0":
            self.update = _support_soft_thresh
            self.c = _process_parameter("c", self.c, 0, len(self.views))
        elif self.regularisation == "l1":
            self.update = _delta_search
            self.c = _process_parameter("c", self.c, 0, len(self.views))
        self.positive = _process_parameter(
            "positive", self.positive, False, len(self.views)
        )

    def _inner_iteration(self):
        c = self.random_state.randn(self.rank)
        c /= np.linalg.norm(c)
        a = self.P @ np.diag(self.D) @ c
        u = self.update(a, self.c[0])
        u /= np.linalg.norm(u)
        b = self.Q @ np.diag(self.D) @ self.P.T @ u
        v = self.update(b, self.c[1])
        v /= np.linalg.norm(v)
        if b.T @ v > self.max_obj:
            self.max_obj = b.T @ v
            self.scores[0] = self.views[0] @ u
            self.scores[1] = self.views[1] @ v
            self.weights[0] = u
            self.weights[1] = v
