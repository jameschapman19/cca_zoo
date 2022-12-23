from itertools import combinations
from typing import Union, Iterable

import numpy as np

from cca_zoo.models._search import _delta_search
from cca_zoo.utils import _process_parameter
from ._base import _BaseIterative
from ._pls_als import _PLSInnerLoop
from .._proximal_operators import support_threshold


class SWCCA(_BaseIterative):
    r"""
    A class used to fit SWCCA model

    References
    ----------
    .. Wenwen, M. I. N., L. I. U. Juan, and Shihua Zhang. "Sparse Weighted Canonical Correlation Analysis." Chinese Journal of Electronics 27.3 (2018): 459-466.

    Examples
    --------
    >>> from cca_zoo.models import SWCCA
    >>> import numpy as np
    >>> rng=np.random.RandomState(0)
    >>> X1 = rng.random((10,5))
    >>> X2 = rng.random((10,5))
    >>> model = SWCCA(regularisation='l0',c=[2, 2], sample_support=5, random_state=0)
    >>> model.fit((X1,X2)).score((X1,X2))
    array([0.61620969])
    """

    def __init__(
        self,
        latent_dims: int = 1,
        scale: bool = True,
        centre=True,
        copy_data=True,
        random_state=None,
        max_iter: int = 500,
        initialization: str = "random",
        tol: float = 1e-9,
        regularisation="l0",
        c: Union[Iterable[Union[float, int]], Union[float, int]] = None,
        sample_support=None,
        positive=False,
        verbose=0,
    ):
        self.c = c
        self.sample_support = sample_support
        self.regularisation = regularisation
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
            verbose=verbose,
        )

    def _set_loop_params(self):
        self.loop = _SWCCAInnerLoop(
            max_iter=self.max_iter,
            tol=self.tol,
            regularisation=self.regularisation,
            c=self.c,
            sample_support=self.sample_support,
            random_state=self.random_state,
            positive=self.positive,
            verbose=self.verbose,
        )

    def _check_params(self):
        if self.sample_support is None:
            self.sample_support = self.n
        self.c = _process_parameter("c", self.c, 2, self.n_views)
        self.positive = _process_parameter(
            "positive", self.positive, False, self.n_views
        )


class _SWCCAInnerLoop(_PLSInnerLoop):
    def __init__(
        self,
        max_iter: int = 100,
        tol=1e-9,
        regularisation="l0",
        c=None,
        sample_support: int = None,
        random_state=None,
        positive=False,
        verbose=0,
    ):
        super().__init__(
            max_iter=max_iter, tol=tol, random_state=random_state, verbose=verbose
        )
        self.c = c
        self.sample_support = sample_support
        if regularisation == "l0":
            self.update = support_threshold
        elif regularisation == "l1":
            self.update = _delta_search
        self.positive = positive

    def _initialize(self, views):
        self.sample_weights = np.ones(self.n)
        self.sample_weights /= np.linalg.norm(self.sample_weights)

    def _update_view(self, views, view_index: int):
        targets = np.ma.array(self.scores, mask=False)
        targets.mask[view_index] = True
        self.weights[view_index] = (
            views[view_index] * self.sample_weights[:, np.newaxis]
        ).T @ targets.sum(axis=0).filled()
        self.weights[view_index] = self.update(
            self.weights[view_index],
            self.c[view_index],
            positive=self.positive[view_index],
        )
        self.weights[view_index] /= np.linalg.norm(self.weights[view_index])
        if view_index == self.n_views - 1:
            self._update_sample_weights()
        self.scores[view_index] = views[view_index] @ self.weights[view_index]

    def _update_sample_weights(self):
        w = self.scores.prod(axis=0)
        self.sample_weights = support_threshold(w, self.sample_support)
        self.sample_weights /= np.linalg.norm(self.sample_weights)
        self.track["sample_weights"] = self.sample_weights

    def _early_stop(self) -> bool:
        return False

    def _objective(self, views) -> int:
        # default objective is correlation
        obj = 0
        for (score_i, score_j) in combinations(self.scores, 2):
            obj += (score_i * self.sample_weights).T @ score_j
        return obj
