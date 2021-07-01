from typing import Iterable

import numpy as np
from scipy.linalg import block_diag, eigh

from .cca_base import _CCA_Base
from ..utils.check_values import _process_parameter


# from hyperopt import fmin, tpe, Trials


class rCCA(_CCA_Base):
    """
    A class used to fit Regularised CCA (canonical ridge) model. Uses PCA to perform the optimization efficiently for high dimensional data.

    Citation
    --------
    Vinod, Hrishikesh D. "Canonical ridge and econometrics of joint production." Journal of econometrics 4.2 (1976): 147-166.

    :Example:

    >>> from cca_zoo.models import rCCA
    >>> X1 = np.random.rand(10,5)
    >>> X2 = np.random.rand(10,5)
    >>> model = rCCA()
    >>> model.fit(X1,X2)
    """

    def __init__(self, latent_dims: int = 1, scale: bool = True, centre=True, copy_data=True, random_state=None,
                 c: Iterable[float] = None,
                 eps=1e-3):
        """
        Constructor for rCCA

        :param latent_dims: number of latent dimensions to fit
        :param scale: normalize variance in each column before fitting
        :param centre: demean data by column before fitting (and before transforming out of sample
        :param copy_data: If True, X will be copied; else, it may be overwritten
        :param random_state: Pass for reproducible output across multiple function calls
        :param c: Iterable of regularisation parameters for each view (between 0:CCA and 1:PLS)
        :param eps: epsilon for stability
        """
        super().__init__(latent_dims=latent_dims, scale=scale, centre=centre, copy_data=copy_data, accept_sparse=True,
                         random_state=random_state)
        self.c = c
        self.eps = eps

    def _check_params(self):
        self.c = _process_parameter('c', self.c, 0, self.n_views)

    def fit(self, *views: np.ndarray):
        """
        Fits a regularised CCA (canonical ridge) model

        :param views: numpy arrays with the same number of rows (samples) separated by commas
        """
        self.n_views = len(views)
        self._check_params()
        train_views = self.centre_scale(*views)
        U_Iterable, S_Iterable, Vt_Iterable = _pca_data(*train_views)
        if len(views) == 2:
            self._two_view_fit(U_Iterable, S_Iterable, Vt_Iterable)
        else:
            self._multi_view_fit(U_Iterable, S_Iterable, Vt_Iterable)
        self.scores = [view @ self.weights[i] for i, view in enumerate(train_views)]
        self.loading_Iterable = [view.T @ score for score, view in zip(self.scores, train_views)]
        self.train_correlations = self.predict_corr(*views)
        return self

    def _two_view_fit(self, U_Iterable, S_Iterable, Vt_Iterable):
        B_Iterable = [(1 - self.c[i]) * S * S + self.c[i] for i, S in
                      enumerate(S_Iterable)]
        R_Iterable = [U @ np.diag(S) for U, S in zip(U_Iterable, S_Iterable)]
        R_12 = R_Iterable[0].T @ R_Iterable[1]
        M = np.diag(1 / np.sqrt(B_Iterable[1])) @ R_12.T @ np.diag(1 / B_Iterable[0]) @ R_12 @ np.diag(
            1 / np.sqrt(B_Iterable[1]))
        n = M.shape[0]
        [eigvals, eigvecs] = eigh(M, subset_by_index=[n - self.latent_dims, n - 1])
        idx = np.argsort(eigvals, axis=0)[::-1]
        eigvecs = eigvecs[:, idx].real
        eigvals = np.real(np.sqrt(eigvals))[idx][:self.latent_dims]
        w_y = Vt_Iterable[1].T @ np.diag(1 / np.sqrt(B_Iterable[1])) @ eigvecs[:, :self.latent_dims].real
        w_x = Vt_Iterable[0].T @ np.diag(1 / B_Iterable[0]) @ R_12 @ np.diag(1 / np.sqrt(B_Iterable[1])) @ eigvecs[:,
                                                                                                           :self.latent_dims].real / eigvals
        self.weights = [w_x, w_y]

    def _multi_view_fit(self, U_Iterable, S_Iterable, Vt_Iterable):
        B_Iterable = [(1 - self.c[i]) * S * S + self.c[i] for i, S in
                      enumerate(S_Iterable)]
        D = block_diag(*[np.diag((1 - self.c[i]) * S * S + self.c[i]) for i, S in
                         enumerate(S_Iterable)])
        C = np.concatenate([U @ np.diag(S) for U, S in zip(U_Iterable, S_Iterable)], axis=1)
        C = C.T @ C
        C -= block_diag(*[np.diag(S ** 2) for U, S in zip(U_Iterable, S_Iterable)]) - D
        D_smallest_eig = min(0, np.linalg.eigvalsh(D).min()) - self.eps
        D = D - D_smallest_eig * np.eye(D.shape[0])
        n = C.shape[0]
        [eigvals, eigvecs] = eigh(C, D, subset_by_index=[n - self.latent_dims, n - 1])
        idx = np.argsort(eigvals, axis=0)[::-1]
        eigvecs = eigvecs[:, idx].real
        splits = np.cumsum([0] + [U.shape[1] for U in U_Iterable])
        self.weights = [Vt.T @ np.diag(1 / np.sqrt(B)) @ eigvecs[split:splits[i + 1], :self.latent_dims] for
                        i, (split, Vt, B) in enumerate(zip(splits[:-1], Vt_Iterable, B_Iterable))]


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

    def __init__(self, latent_dims: int = 1, scale: bool = True, centre=True, copy_data=True, random_state=None):
        """
        Constructor for CCA
        :param latent_dims: number of latent dimensions to fit
        :param scale: normalize variance in each column before fitting
        :param centre: demean data by column before fitting (and before transforming out of sample
        :param copy_data: If True, X will be copied; else, it may be overwritten
        :param random_state: Pass for reproducible output across multiple function calls
        """
        super().__init__(latent_dims=latent_dims, scale=scale, centre=centre, copy_data=copy_data, c=[0.0, 0.0],
                         random_state=random_state)


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
