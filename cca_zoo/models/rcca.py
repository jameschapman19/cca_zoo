from abc import abstractmethod
from typing import Iterable, Union

import numpy as np
from scipy.linalg import block_diag, eigh

from cca_zoo.models.cca_base import _CCA_Base
from cca_zoo.utils.check_values import _process_parameter, check_views


# from hyperopt import fmin, tpe, Trials


class rCCA(_CCA_Base):
    r"""
    A class used to fit Regularised CCA (canonical ridge) model. Uses PCA to perform the optimization efficiently for high dimensional data.

    :Maths:

    .. math::

        w_{opt}=\underset{w}{\mathrm{argmax}}\{ w_1^TX_1^TX_2w_2  \}\\

        \text{subject to:}

        (1-c_1)w_1^TX_1^TX_1w_1+c_1w_1^Tw_1=1

        (1-c_2)w_2^TX_2^TX_2w_2+c_2w_2^Tw_2=1


    :Citation:

    Vinod, Hrishikesh D. "Canonical ridge and econometrics of joint production." Journal of econometrics 4.2 (1976): 147-166.

    :Example:

    >>> from cca_zoo.models import rCCA
    >>> import numpy as np
    >>> rng=np.random.RandomState(0)
    >>> X1 = rng.random((10,5))
    >>> X2 = rng.random((10,5))
    >>> model = rCCA(c=[0.1,0.1])
    >>> model.fit((X1,X2)).score((X1,X2))
    array([0.95222128])
    """

    def __init__(
        self,
        latent_dims: int = 1,
        scale: bool = True,
        centre=True,
        copy_data=True,
        random_state=None,
        c: Union[Iterable[float], float] = None,
        eps=1e-3,
        accept_sparse=None,
    ):
        """
        Constructor for rCCA

        :param latent_dims: number of latent dimensions to fit
        :param scale: normalize variance in each column before fitting
        :param centre: demean data by column before fitting (and before transforming out of sample
        :param copy_data: If True, X will be copied; else, it may be overwritten
        :param random_state: Pass for reproducible output across multiple function calls
        :param c: Iterable of regularisation parameters for each view (between 0:CCA and 1:PLS)
        :param eps: epsilon for stability
        :param accept_sparse: which forms are accepted for sparse data
        """
        if accept_sparse is None:
            accept_sparse = ["csc", "csr"]
        super().__init__(
            latent_dims=latent_dims,
            scale=scale,
            centre=centre,
            copy_data=copy_data,
            accept_sparse=accept_sparse,
            random_state=random_state,
        )
        self.c = c
        self.eps = eps

    def _check_params(self):
        self.c = _process_parameter("c", self.c, 0, self.n_views)

    def fit(self, views: Iterable[np.ndarray], y=None, **kwargs):
        """
        Fits a regularised CCA (canonical ridge) model

        :param views: list/tuple of numpy arrays or array likes with the same number of rows (samples)
        """
        views = check_views(
            *views, copy=self.copy_data, accept_sparse=self.accept_sparse
        )
        views = self._centre_scale(views)
        self.n_views = len(views)
        self.n = views[0].shape[0]
        self._check_params()
        views, C, D = self._setup_evp(views, **kwargs)
        self._solve_evp(views, C, D, **kwargs)
        return self

    @abstractmethod
    def _setup_evp(self, views: Iterable[np.ndarray], **kwargs):
        Us, Ss, Vts = _pca_data(*views)
        self.Bs = [
            (1 - self.c[i]) * S * S / self.n + self.c[i] for i, S in enumerate(Ss)
        ]
        if len(views) == 2:
            self._two_view = True
            C, D = self._two_view_evp(Us, Ss)
        else:
            self._two_view = False
            C, D = self._multi_view_evp(Us, Ss)
        return Vts, C, D

    @abstractmethod
    def _solve_evp(self, views: Iterable[np.ndarray], C, D=None, **kwargs):
        p = C.shape[0]
        if self._two_view:
            [eigvals, eigvecs] = eigh(C, subset_by_index=[p - self.latent_dims, p - 1])
            eigvecs = eigvecs
            idx = np.argsort(eigvals, axis=0)[::-1][: self.latent_dims]
            eigvecs = eigvecs[:, idx].real
            w_y = views[1].T @ np.diag(1 / np.sqrt(self.Bs[1])) @ eigvecs
            w_x = (
                views[0].T
                @ np.diag(1 / self.Bs[0])
                @ self.R_12
                @ np.diag(1 / np.sqrt(self.Bs[1]))
                @ eigvecs
                / np.sqrt(eigvals[idx])
            )
            self.weights = [w_x, w_y]
        else:
            [eigvals, eigvecs] = eigh(
                C, D, subset_by_index=[p - self.latent_dims, p - 1]
            )
            idx = np.argsort(eigvals, axis=0)[::-1]
            eigvecs = eigvecs[:, idx].real
            self.weights = [
                Vt.T
                @ np.diag(1 / np.sqrt(B))
                @ eigvecs[split : self.splits[i + 1], : self.latent_dims]
                for i, (split, Vt, B) in enumerate(
                    zip(self.splits[:-1], views, self.Bs)
                )
            ]

    def _two_view_evp(self, Us, Ss):
        Rs = [U @ np.diag(S) for U, S in zip(Us, Ss)]
        self.R_12 = Rs[0].T @ Rs[1]
        M = (
            np.diag(1 / np.sqrt(self.Bs[1]))
            @ self.R_12.T
            @ np.diag(1 / self.Bs[0])
            @ self.R_12
            @ np.diag(1 / np.sqrt(self.Bs[1]))
        )
        return M, None

    def _multi_view_evp(self, Us, Ss):
        D = block_diag(
            *[np.diag((1 - self.c[i]) * S * S + self.c[i]) for i, S in enumerate(Ss)]
        )
        C = np.concatenate([U @ np.diag(S) for U, S in zip(Us, Ss)], axis=1)
        C = C.T @ C
        C -= block_diag(*[np.diag(S ** 2) for U, S in zip(Us, Ss)]) - D
        D_smallest_eig = min(0, np.linalg.eigvalsh(D).min()) - self.eps
        D = D - D_smallest_eig * np.eye(D.shape[0])
        self.splits = np.cumsum([0] + [U.shape[1] for U in Us])
        return C, D


class CCA(rCCA):
    r"""
    A class used to fit a simple CCA model

    Implements CCA by inheriting regularised CCA with 0 regularisation

    :Maths:

    .. math::

        w_{opt}=\underset{w}{\mathrm{argmax}}\{ w_1^TX_1^TX_2w_2  \}\\

        \text{subject to:}

        w_1^TX_1^TX_1w_1=1

        w_2^TX_2^TX_2w_2=1

    :Citation:

    Hotelling, Harold. "Relations between two sets of variates." Breakthroughs in statistics. Springer, New York, NY, 1992. 162-190.

    :Example:

    >>> from cca_zoo.models import CCA
    >>> import numpy as np
    >>> rng=np.random.RandomState(0)
    >>> X1 = rng.random((10,5))
    >>> X2 = rng.random((10,5))
    >>> model = CCA()
    >>> model.fit((X1,X2)).score((X1,X2))
    array([1.])
    """

    def __init__(
        self,
        latent_dims: int = 1,
        scale: bool = True,
        centre=True,
        copy_data=True,
        random_state=None,
    ):
        """
        Constructor for CCA

        :param latent_dims: number of latent dimensions to fit
        :param scale: normalize variance in each column before fitting
        :param centre: demean data by column before fitting (and before transforming out of sample
        :param copy_data: If True, X will be copied; else, it may be overwritten
        :param random_state: Pass for reproducible output across multiple function calls
        """
        super().__init__(
            latent_dims=latent_dims,
            scale=scale,
            centre=centre,
            copy_data=copy_data,
            c=[0.0, 0.0],
            random_state=random_state,
        )


class PLS(rCCA):
    r"""
    A class used to fit a simple PLS model

    Implements PLS by inheriting regularised CCA with maximal regularisation

    :Maths:

    .. math::

        w_{opt}=\underset{w}{\mathrm{argmax}}\{ w_1^TX_1^TX_2w_2  \}\\

        \text{subject to:}

        w_1^Tw_1=1

        w_2^Tw_2=1

    :Example:

    >>> from cca_zoo.models import PLS
    >>> import numpy as np
    >>> rng=np.random.RandomState(0)
    >>> X1 = rng.random((10,5))
    >>> X2 = rng.random((10,5))
    >>> model = PLS()
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
    ):
        """
        Constructor for PLS

        :param latent_dims: number of latent dimensions to fit
        :param scale: normalize variance in each column before fitting
        :param centre: demean data by column before fitting (and before transforming out of sample
        :param copy_data: If True, X will be copied; else, it may be overwritten
        :param random_state: Pass for reproducible output across multiple function calls
        """
        super().__init__(
            latent_dims=latent_dims,
            scale=scale,
            centre=centre,
            copy_data=copy_data,
            c=[1.0, 1.0],
            random_state=random_state,
        )


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
