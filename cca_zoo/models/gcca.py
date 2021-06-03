from typing import Tuple, List

import numpy as np
from scipy.linalg import eigh

from .cca_base import _CCA_Base
from ..utils.check_values import _process_parameter


class GCCA(_CCA_Base):
    """
    A class used to fit GCCA model. For more than 2 views, GCCA optimizes the sum of correlations with a shared auxiliary vector

    :Example:

    >>> from cca_zoo.models import GCCA
    >>> X1 = np.random.rand(10,5)
    >>> X2 = np.random.rand(10,5)
    >>> model = GCCA()
    >>> model.fit(X1,X2)
    """

    def __init__(self, latent_dims: int = 1, c: List[float] = None, view_weights: Tuple[float, ...] = None, scale=True):
        """
        Constructor for GCCA

        :param latent_dims: number of latent dimensions
        :param c: regularisation between 0 (CCA) and 1 (PLS)
        :param view_weights: list of weights of each view
        """
        super().__init__(latent_dims=latent_dims, scale=scale)
        self.c = c
        self.view_weights = view_weights

    def check_params(self):
        self.c = _process_parameter('c', self.c, 0, self.n_views)
        self.view_weights = _process_parameter('view_weights', self.view_weights, 1, self.n_views)

    def fit(self, *views: np.ndarray, K: np.ndarray = None):
        """
        Fits a GCCA model

        :param views: numpy arrays with the same number of rows (samples) separated by commas
        :param K: observation matrix. Binary array with (k,n) dimensions where k is the number of views and n is the number of samples 1 means the data is observed in the corresponding view and 0 means the data is unobserved in that view.
        """

        self.n_views = len(views)
        self.check_params()

        if K is None:
            # just use identity when all rows are observed in all views.
            K = np.ones((len(views), views[0].shape[0]))
        train_views = self.centre_scale(*views)
        Q = []
        for i, (view, view_weight) in enumerate(zip(train_views, self.view_weights)):
            view_cov = view.T @ view
            view_cov = (1 - self.c[i]) * view_cov + self.c[i] * np.eye(view_cov.shape[0])
            Q.append(view_weight * view @ np.linalg.inv(view_cov) @ view.T)
        Q = np.sum(Q, axis=0)
        Q = np.diag(np.sqrt(np.sum(K, axis=0))) @ Q @ np.diag(np.sqrt(np.sum(K, axis=0)))
        n = Q.shape[0]
        [eigvals, eigvecs] = eigh(Q, subset_by_index=[n - self.latent_dims, n - 1])
        idx = np.argsort(eigvals, axis=0)[::-1]
        eigvecs = eigvecs[:, idx].real
        self.eigvals = eigvals[idx].real
        self.weights_list = [np.linalg.pinv(view) @ eigvecs[:, :self.latent_dims] for view in train_views]
        self.score_list = [view @ self.weights_list[i] for i, view in enumerate(train_views)]
        self.train_correlations = self.predict_corr(*views)
        return self

    def transform(self, *views: np.ndarray, K=None, view_indices: List[int] = None, **kwargs):
        """
        Transforms data given a fit GCCA model

        :param views: numpy arrays with the same number of rows (samples) separated by commas
        :param K: observation matrix. Binary array with (k,n) dimensions where k is the number of views and n is the number of samples
        1 means the data is observed in the corresponding view and 0 means the data is unobserved in that view.
        """
        transformed_views = []
        if view_indices is None:
            view_indices = np.arange(len(views))
        for i, (view, view_index) in enumerate(zip(views, view_indices)):
            view = view.copy(order='K')
            transformed_view = np.array((view - self.view_means[view_index]) @ self.weights_list[view_index])
            # TODO maybe revisit this. The original idea was to only generate correlations for observed samples but it's perhaps simpler to do this in post processing
            # if K is not None:
            #    transformed_view.mask[np.where(K[view_index]) == 1] = True
            transformed_views.append(transformed_view)
        return transformed_views
