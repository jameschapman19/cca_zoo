from typing import List

import numpy as np
from scipy.linalg import block_diag, eigh

from .cca_base import _CCA_Base
from ..utils.check_values import _process_parameter


# from hyperopt import fmin, tpe, Trials


class rCCA(_CCA_Base):
    """
    A class used to fit Regularised CCA (canonical ridge) model. Uses PCA to perform the optimization efficiently for high dimensional data.

    :Example:

    >>> from cca_zoo.models import rCCA
    >>> X1 = np.random.rand(10,5)
    >>> X2 = np.random.rand(10,5)
    >>> model = rCCA()
    >>> model.fit(X1,X2)
    """

    def __init__(self, latent_dims: int = 1, c: List[float] = None, scale=True):
        """
        Constructor for rCCA

        :param latent_dims: number of latent dimensions
        :param c: regularisation between 0 (CCA) and 1 (PLS)
        """
        super().__init__(latent_dims=latent_dims, scale=scale)
        self.c = c

    def check_params(self):
        self.c = _process_parameter('c', self.c, 0, self.n_views)

    def fit(self, *views: np.ndarray):
        """
        Fits a regularised CCA (canonical ridge) model

        :param views: numpy arrays with the same number of rows (samples) separated by commas
        """
        self.n_views = len(views)
        self.check_params()
        train_views = self.centre_scale(*views)
        U_list, S_list, Vt_list = _pca_data(*train_views)
        if len(views) == 2:
            self.two_view_fit(U_list, S_list, Vt_list)
        else:
            self.multi_view_fit(U_list, S_list, Vt_list)
        self.score_list = [view @ self.weights_list[i] for i, view in enumerate(train_views)]
        self.loading_list = [view.T @ score for score, view in zip(self.score_list, train_views)]
        self.train_correlations = self.predict_corr(*views)
        return self

    def two_view_fit(self, U_list, S_list, Vt_list):
        B_list = [(1 - self.c[i]) * S * S + self.c[i] for i, S in
                  enumerate(S_list)]
        R_list = [U @ np.diag(S) for U, S in zip(U_list, S_list)]
        R_12 = R_list[0].T @ R_list[1]
        M = np.diag(1 / np.sqrt(B_list[1])) @ R_12.T @ np.diag(1 / B_list[0]) @ R_12 @ np.diag(1 / np.sqrt(B_list[1]))
        n = M.shape[0]
        [eigvals, eigvecs] = eigh(M, subset_by_index=[n - self.latent_dims, n - 1])
        idx = np.argsort(eigvals, axis=0)[::-1]
        eigvecs = eigvecs[:, idx].real
        eigvals = np.real(np.sqrt(eigvals))[idx][:self.latent_dims]
        w_y = Vt_list[1].T @ np.diag(1 / np.sqrt(B_list[1])) @ eigvecs[:, :self.latent_dims].real
        w_x = Vt_list[0].T @ np.diag(1 / B_list[0]) @ R_12 @ np.diag(1 / np.sqrt(B_list[1])) @ eigvecs[:,
                                                                                               :self.latent_dims].real / eigvals
        self.weights_list = [w_x, w_y]

    def multi_view_fit(self, U_list, S_list, Vt_list):
        B_list = [(1 - self.c[i]) * S * S + self.c[i] for i, S in
                  enumerate(S_list)]
        D = block_diag(*[np.diag((1 - self.c[i]) * S * S + self.c[i]) for i, S in
                         enumerate(S_list)])
        C = np.concatenate([U @ np.diag(S) for U, S in zip(U_list, S_list)], axis=1)
        C = C.T @ C
        C -= block_diag(*[np.diag(S ** 2) for U, S in zip(U_list, S_list)]) - D
        n = C.shape[0]
        [eigvals, eigvecs] = eigh(C, D, subset_by_index=[n - self.latent_dims, n - 1])
        idx = np.argsort(eigvals, axis=0)[::-1]
        eigvecs = eigvecs[:, idx].real
        splits = np.cumsum([0] + [U.shape[1] for U in U_list])
        self.weights_list = [Vt.T @ np.diag(1 / np.sqrt(B)) @ eigvecs[split:splits[i + 1], :self.latent_dims] for
                             i, (split, Vt, B) in enumerate(zip(splits[:-1], Vt_list, B_list))]


class CCA(rCCA):
    """
    A class used to fit a simple CCA model

    Implements CCA by inheriting regularised CCA with 0 regularisation

    :Example:

    >>> from cca_zoo.models import CCA
    >>> X1 = np.random.rand(10,5)
    >>> X2 = np.random.rand(10,5)
    >>> model = CCA()
    >>> model.fit(X1,X2)
    """

    def __init__(self, latent_dims: int = 1, scale=True):
        """
        Constructor for CCA

        :param latent_dims: number of latent dimensions to learn
        """
        super().__init__(latent_dims=latent_dims, c=[0.0, 0.0], scale=scale)


def _pca_data(*views: np.ndarray):
    """
    :param views: numpy arrays with the same number of rows (samples) separated by commas
    """
    views_U = []
    views_S = []
    views_Vt = []
    for i, view in enumerate(views):
        U, S, Vt = np.linalg.svd(view, full_matrices=False)
        views_U.append(U)
        views_S.append(S)
        views_Vt.append(Vt)
    return views_U, views_S, views_Vt
