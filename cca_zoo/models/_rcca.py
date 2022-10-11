from typing import Iterable, Union

import numpy as np
from scipy.linalg import block_diag, eigh
from sklearn.decomposition import PCA
from cca_zoo.utils.check_values import _process_parameter
from ._base import _BaseCCA


class rCCA(_BaseCCA):
    r"""
    A class used to fit Regularised CCA (canonical ridge) model. Uses PCA to perform the optimization efficiently for high dimensional data.

    .. math::

        w_{opt}=\underset{w}{\mathrm{argmax}}\{ w_1^TX_1^TX_2w_2  \}\\

        \text{subject to:}

        (1-c_1)w_1^TX_1^TX_1w_1+c_1w_1^Tw_1=n

        (1-c_2)w_2^TX_2^TX_2w_2+c_2w_2^Tw_2=n

    Parameters
    ----------
    latent_dims : int, optional
        Number of latent dimensions to use, by default 1
    scale : bool, optional
        Whether to scale the data, by default True
    centre : bool, optional
        Whether to centre the data, by default True
    copy_data : bool, optional
        Whether to copy the data, by default True
    random_state : int, optional
        Random state for reproducibility, by default None
    c : Union[Iterable[float], float], optional
        Regularisation parameter, by default None
    eps : float, optional
        Tolerance for convergence, by default 1e-3
    accept_sparse : Union[Iterable[str], str], optional
        Whether to accept sparse matrices, by default None

    References
    --------
    Vinod, Hrishikesh D. "Canonical ridge and econometrics of joint production." Journal of econometrics 4.2 (1976): 147-166.

    Example
    -------
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
        views = self._validate_inputs(views)
        self._check_params()
        C, D = self._setup_evp(views, **kwargs)
        eigvals, eigvecs = self._solve_evp(C, D, **kwargs)
        self._get_weights(eigvals, eigvecs, views)
        return self

    def _get_weights(self, eigvals, eigvecs, views):
        if self._two_view:
            w_y = self.principal_components[1].components_.T @ np.diag(1 / np.sqrt(self.B[1])) @ eigvecs
            w_x = (
                    self.principal_components[0].components_.T
                    @ np.diag(1 / self.B[0])
                    @ self.principal_components[0].transform(views[0]).T @ self.principal_components[1].transform(
                views[1])
                    @ eigvecs
                    / np.sqrt(eigvals)
            )
            self.weights = [w_x, w_y]
        else:
            self.weights = [
                self.principal_components[i].components_.T
                @ np.diag(1 / np.sqrt(B))
                @ eigvecs[split: self.splits[i + 1], : self.latent_dims]
                for i, (split, B) in enumerate(
                    zip(self.splits[:-1], self.B)
                )
            ]

    def _setup_evp(self, views: Iterable[np.ndarray], **kwargs):
        self.principal_components = _pca_data(views)
        S = [pc.singular_values_ for pc in self.principal_components]
        self.B = [(1 - self.c[i]) * S ** 2 + self.c[i] for i, S in enumerate(S)]
        if self.n_views == 2:
            self._two_view = True
            C, D = self._two_view_evp(views)
        else:
            self._two_view = False
            C, D = self._multi_view_evp(views)
        return C, D

    def _solve_evp(self, C, D=None):
        p = C.shape[0]
        [eigvals, eigvecs] = eigh(C, D, subset_by_index=[p - self.latent_dims, p - 1])
        idx = np.argsort(eigvals, axis=0)[::-1][: self.latent_dims]
        eigvecs = eigvecs[:, idx].real
        return eigvals, eigvecs

    def _two_view_evp(self, views):
        R = [pca.transform(view) for pca, view in zip(self.principal_components, views)]
        C = np.cov(R[1], R[0],rowvar=False) @ np.cov(R[0], R[1],rowvar=False)
        return C, None

    def _multi_view_evp(self, views):
        R = [pca.transform(view) for pca, view in zip(self.principal_components, views)]
        D = block_diag(
            *[np.diag(B) for B in self.B]
        )
        C = np.cov(np.hstack(R),rowvar=False)
        C -= block_diag(*[np.cov(R_,rowvar=False) for R_ in R])
        D_smallest_eig = min(0, np.linalg.eigvalsh(D).min()) - self.eps
        D = D - D_smallest_eig * np.eye(D.shape[0])
        self.splits = np.cumsum([0] + [R_.shape[1] for R_ in R])
        return C, D


class CCA(rCCA):
    r"""
    A class used to fit a simple CCA model

    .. math::

        w_{opt}=\underset{w}{\mathrm{argmax}}\{ w_1^TX_1^TX_2w_2  \}\\

        \text{subject to:}

        w_1^TX_1^TX_1w_1=n

        w_2^TX_2^TX_2w_2=n

    Parameters
    ----------
    latent_dims : int, optional
        The number of latent dimensions to use, by default 1
    scale : bool, optional
        Whether to scale the data, by default True
    centre : bool, optional
        Whether to centre the data, by default True
    copy_data : bool, optional
        Whether to copy the data, by default True
    random_state : int, optional
        The random state to use, by default None

    References
    --------

    Hotelling, Harold. "Relations between two sets of variates." Breakthroughs in statistics. Springer, New York, NY, 1992. 162-190.

    Example
    -------
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
        super().__init__(
            latent_dims=latent_dims,
            scale=scale,
            centre=centre,
            copy_data=copy_data,
            c=0.0,
            random_state=random_state,
        )


class PLS(rCCA):
    r"""
    A class used to fit a simple PLS model

    Implements PLS by inheriting regularised CCA with maximal regularisation

    .. math::

        w_{opt}=\underset{w}{\mathrm{argmax}}\{ w_1^TX_1^TX_2w_2  \}\\

        \text{subject to:}

        w_1^Tw_1=1

        w_2^Tw_2=1

    Example
    -------

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
        super().__init__(
            latent_dims=latent_dims,
            scale=scale,
            centre=centre,
            copy_data=copy_data,
            c=1,
            random_state=random_state,
        )


def _pca_data(views: Iterable[np.ndarray]):
    """
    Performs PCA on the data



    """
    PC = []
    for i, view in enumerate(views):
        PC.append(PCA(whiten=True).fit(view))
    return PC
