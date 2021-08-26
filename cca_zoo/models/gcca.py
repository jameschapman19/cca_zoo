from typing import Iterable

import numpy as np
from scipy.linalg import eigh

from .cca_base import _CCA_Base
from ..utils.check_values import _process_parameter, check_views


class GCCA(_CCA_Base):
    """
    A class used to fit GCCA model. For more than 2 views, GCCA optimizes the sum of correlations with a shared auxiliary vector

    Citation
    --------
    Tenenhaus, Arthur, and Michel Tenenhaus. "Regularized generalized canonical correlation analysis." Psychometrika 76.2 (2011): 257.

    :Example:

    >>> from cca_zoo.models import GCCA
    >>> rng=np.random.RandomState(0)
    >>> X1 = rng.random(10,5)
    >>> X2 = np.random.rand(10,5)
    >>> model = GCCA()
    >>> model.fit(X1,X2)
    """

    def __init__(self, latent_dims: int = 1, scale: bool = True, centre=True, copy_data=True, random_state=None,
                 c: Iterable[float] = None,
                 view_weights: Iterable[float] = None):
        """
        Constructor for GCCA

        :param latent_dims: number of latent dimensions to fit
        :param scale: normalize variance in each column before fitting
        :param centre: demean data by column before fitting (and before transforming out of sample
        :param copy_data: If True, X will be copied; else, it may be overwritten
        :param random_state: Pass for reproducible output across multiple function calls
        :param c: regularisation between 0 (CCA) and 1 (PLS)
        :param view_weights: list of weights of each view
        """
        super().__init__(latent_dims=latent_dims, scale=scale, centre=centre, copy_data=copy_data,
                         accept_sparse=['csc', 'csr'],
                         random_state=random_state)
        self.c = c
        self.view_weights = view_weights

    def _check_params(self):
        self.c = _process_parameter('c', self.c, 0, self.n_views)
        self.view_weights = _process_parameter('view_weights', self.view_weights, 1, self.n_views)

    def fit(self, *views: np.ndarray, K: np.ndarray = None):
        """
        Fits a GCCA model

        :param views: numpy arrays with the same number of rows (samples) separated by commas
        :param K: observation matrix. Binary array with (k,n) dimensions where k is the number of views and n is the number of samples 1 means the data is observed in the corresponding view and 0 means the data is unobserved in that view.
        """
        views = check_views(*views, copy=self.copy_data, accept_sparse=self.accept_sparse)
        views = self._centre_scale(*views)
        self.n_views = len(views)
        self.n = views[0].shape[0]
        self._check_params()
        if K is None:
            # just use identity when all rows are observed in all views.
            K = np.ones((len(views), views[0].shape[0]))
        Q = []
        for i, (view, view_weight) in enumerate(zip(views, self.view_weights)):
            view_cov = view.T @ view / self.n
            view_cov = (1 - self.c[i]) * view_cov + self.c[i] * np.eye(view_cov.shape[0])
            Q.append(view_weight * view @ np.linalg.inv(view_cov) @ view.T)
        Q = np.sum(Q, axis=0)
        Q = np.diag(np.sqrt(np.sum(K, axis=0))) @ Q @ np.diag(np.sqrt(np.sum(K, axis=0)))
        n = Q.shape[0]
        [eigvals, eigvecs] = eigh(Q, subset_by_index=[n - self.latent_dims, n - 1])
        idx = np.argsort(eigvals, axis=0)[::-1][:self.latent_dims]
        eigvecs = eigvecs[:, idx].real
        self.weights = [np.linalg.pinv(view) @ eigvecs[:, :self.latent_dims] for view in views]
        self.scores = [view @ self.weights[i] for i, view in enumerate(views)]
        return self
