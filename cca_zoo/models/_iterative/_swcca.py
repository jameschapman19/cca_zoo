from itertools import combinations
from typing import Union, Iterable

import numpy as np

from cca_zoo.models._iterative._base import _BaseIterative
from cca_zoo.models._search import _delta_search, support_threshold
from cca_zoo.utils import _process_parameter


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
    >>> model = SWCCA(regularisation='l0',tau=[2, 2], sample_support=5, random_state=0)
    >>> model.fit((X1,X2)).score((X1,X2))
    array([0.61620969])
    """

    def __init__(
        self,
        scale: bool = True,
        centre=True,
        copy_data=True,
        random_state=None,
        max_iter: int = 500,
        initialization: str = "random",
        tol: float = 1e-3,
        regularisation="l0",
        tau: Union[Iterable[Union[float, int]], Union[float, int]] = None,
        sample_support=None,
        positive=False,
        verbose=0,
    ):
        self.tau = tau
        self.sample_support = sample_support
        if regularisation == "l0":
            self.update = support_threshold
        elif regularisation == "l1":
            self.update = _delta_search
        self.regularisation = regularisation
        self.positive = positive
        super().__init__(
            latent_dims=1,
            scale=scale,
            centre=centre,
            copy_data=copy_data,
            max_iter=max_iter,
            initialization=initialization,
            tol=tol,
            random_state=random_state,
            verbose=verbose,
        )

    def _check_params(self):
        if self.sample_support is None:
            self.sample_support = self.n
        self.tau = _process_parameter("tau", self.tau, 2, self.n_views)
        self.positive = _process_parameter(
            "positive", self.positive, False, self.n_views
        )

    def _initialize(self, views):
        self.sample_weights = np.ones(self.n)
        self.sample_weights /= np.linalg.norm(self.sample_weights)

    def _update(self, views, scores, weights):
        # Update each view using loop update function
        for view_index, view in enumerate(views):
            targets = np.ma.array(scores, mask=False)
            targets.mask[view_index] = True
            weights[view_index] = (
                views[view_index] * self.sample_weights[:, np.newaxis]
            ).T @ targets.sum(axis=0).filled()
            weights[view_index] = self.update(
                weights[view_index],
                self.tau[view_index],
                positive=self.positive[view_index],
            )
            weights[view_index] /= np.linalg.norm(weights[view_index])
            if view_index == self.n_views - 1:
                self.sample_weights = self._update_sample_weights(scores)
            scores[view_index] = views[view_index] @ weights[view_index]
        return scores, weights

    def _update_sample_weights(self, scores):
        w = scores.prod(axis=0)
        sample_weights = support_threshold(w, self.sample_support)
        sample_weights /= np.linalg.norm(sample_weights)
        return sample_weights

    def _objective(self, views, scores, weights) -> int:
        # default objective is correlation
        obj = 0
        for (score_i, score_j) in combinations(scores, 2):
            obj += (score_i * self.sample_weights).T @ score_j
        return obj
