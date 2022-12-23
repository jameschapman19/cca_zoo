from typing import Union, Iterable

import numpy as np
from sklearn.metrics import pairwise_kernels
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_is_fitted

from cca_zoo.models._base import _BaseCCA
from cca_zoo.utils.check_values import _process_parameter, _check_views


class NCCA(_BaseCCA):
    """
    A class used to fit nonparametric (NCCA) model.

    References
    ----------
    Michaeli, Tomer, Weiran Wang, and Karen Livescu. "Nonparametric canonical correlation analysis." International conference on machine learning. PMLR, 2016.

    Example
    -------
    >>> from cca_zoo.models import NCCA
    >>> X1 = np.random.rand(10,5)
    >>> X2 = np.random.rand(10,5)
    >>> model = NCCA()
    >>> model._fit((X1,X2)).score((X1,X2))
    array([1.])
    """

    def __init__(
        self,
        latent_dims: int = 1,
        scale=True,
        centre=True,
        copy_data=True,
        accept_sparse=False,
        random_state: Union[int, np.random.RandomState] = None,
        nearest_neighbors=None,
        gamma: Iterable[float] = None,
    ):
        super().__init__(
            latent_dims, scale, centre, copy_data, accept_sparse, random_state
        )
        self.nearest_neighbors = nearest_neighbors
        self.gamma = gamma

    def _check_params(self):
        self.nearest_neighbors = _process_parameter(
            "nearest_neighbors", self.nearest_neighbors, 1, self.n_views
        )
        self.gamma = _process_parameter("gamma", self.gamma, None, self.n_views)
        self.kernel = _process_parameter("kernel", None, "rbf", self.n_views)

    def fit(self, views: Iterable[np.ndarray], y=None, **kwargs):
        views = self._validate_inputs(views)
        self._check_params()
        self.train_views = views
        self.knns = [
            NearestNeighbors(n_neighbors=self.nearest_neighbors[i]).fit(view)
            for i, view in enumerate(views)
        ]
        NNs = [
            self.knns[i].kneighbors(view, self.nearest_neighbors[i])
            for i, view in enumerate(views)
        ]
        kernels = [self._get_kernel(i, view) for i, view in enumerate(self.train_views)]
        self.Ws = [fill_w(kernel, inds) for kernel, (dists, inds) in zip(kernels, NNs)]
        self.Ws = [
            self.Ws[0] / self.Ws[0].sum(axis=1, keepdims=True),
            self.Ws[1] / self.Ws[1].sum(axis=0, keepdims=True),
        ]
        S = self.Ws[0] @ self.Ws[1]
        U, S, Vt = np.linalg.svd(S)
        self.f = U[:, 1 : self.latent_dims + 1] * np.sqrt(self.n)
        self.g = Vt[1 : self.latent_dims + 1, :].T * np.sqrt(self.n)
        self.S = S[1 : self.latent_dims + 1]
        return self

    def transform(self, views: Iterable[np.ndarray], **kwargs):
        check_is_fitted(self, attributes=["f", "g"])
        views = _check_views(
            *views, copy=self.copy_data, accept_sparse=self.accept_sparse
        )
        views = self._centre_scale_transform(views)
        nns = [
            self.knns[i].kneighbors(view, self.nearest_neighbors[i])
            for i, view in enumerate(views)
        ]
        kernels = [
            self._get_kernel(i, self.train_views[i], Y=view)
            for i, view in enumerate(views)
        ]
        Wst = [fill_w(kernel, inds) for kernel, (dists, inds) in zip(kernels, nns)]
        Wst = [
            Wst[0] / Wst[0].sum(axis=1, keepdims=True),
            Wst[1] / Wst[1].sum(axis=1, keepdims=True),
        ]
        St = [Wst[0] @ self.Ws[1], Wst[1] @ self.Ws[0]]
        return St[0] @ self.g * (1 / self.S), St[1] @ self.f * (1 / self.S)

    def _get_kernel(self, view, X, Y=None):
        params = {
            "gamma": self.gamma[view],
        }
        return pairwise_kernels(
            X, Y, metric=self.kernel[view], filter_params=True, **params
        )


def fill_w(kernels, inds):
    w = np.zeros_like(kernels)
    for i, ind in enumerate(inds):
        w[ind, i] = kernels[ind, i]
    return w.T
