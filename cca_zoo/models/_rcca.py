from typing import Iterable, Union

import numpy as np
from scipy.linalg import eigh
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
        Random state, by default None
    c : Union[Iterable[float], float], optional
        Regularisation parameter, by default None
    accept_sparse : Union[bool, str], optional
        Whether to accept sparse data, by default None


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

    def _check_params(self):
        self.c = _process_parameter("c", self.c, 0, self.n_views)

    def fit(self, views: Iterable[np.ndarray], y=None, **kwargs):
        views = self._validate_inputs(views)
        self._check_params()
        C, D = self._setup_evp(views, **kwargs)
        eigvals, eigvecs = self._solve_evp(C, D)
        self._weights(eigvals, eigvecs, views)
        return self

    def _setup_evp(self, views: Iterable[np.ndarray], **kwargs):
        n = views[0].shape[0]
        self.principal_components = _pca_data(*views)
        self.Bs = [
            (1 - self.c[i]) * pc.singular_values_**2 / n + self.c[i]
            for i, pc in enumerate(self.principal_components)
        ]
        C, D = self._two_view_evp(views)
        return C, D

    def _weights(self, eigvals, eigvecs, views):
        R, B = self._get_R_B(views)
        R_12 = R[0].T @ R[1]
        w_y = (
            self.principal_components[1].components_.T
            @ np.diag(1 / np.sqrt(B[1]))
            @ eigvecs
        )
        w_x = (
            self.principal_components[0].components_.T
            @ np.diag(1 / B[0])
            @ R_12
            @ np.diag(1 / np.sqrt(B[1]))
            @ eigvecs
            / np.sqrt(eigvals)
        )
        self.weights = [w_x, w_y]

    def _get_R_B(self, views):
        n = views[0].shape[0]
        B = [
            (1 - self.c[i]) * pc.singular_values_**2 / n + self.c[i]
            for i, pc in enumerate(self.principal_components)
        ]
        R = [pc.transform(view) for view, pc in zip(views, self.principal_components)]
        return R, B

    def _solve_evp(self, C, D=None):
        p = C.shape[0]
        [eigvals, eigvecs] = eigh(
            C,
            D,
            subset_by_index=[p - self.latent_dims, p - 1],
        )
        idx = np.argsort(eigvals, axis=0)[::-1]
        eigvecs = eigvecs[:, idx].real
        return np.flip(eigvals), eigvecs

    def _two_view_evp(self, views):
        R, B = self._get_R_B(views)
        R_12 = R[0].T @ R[1]
        M = (
            np.diag(1 / np.sqrt(B[1]))
            @ R_12.T
            @ np.diag(1 / B[0])
            @ R_12
            @ np.diag(1 / np.sqrt(B[1]))
        )
        return M, None

    def _more_tags(self):
        return {"multiview": True}


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
        Number of latent dimensions to use, by default 1
    scale : bool, optional
        Whether to scale the data, by default True
    centre : bool, optional
        Whether to centre the data, by default True
    copy_data : bool, optional
        Whether to copy the data, by default True
    random_state : int, optional
        Random state, by default None
    accept_sparse : Union[bool, str], optional
        Whether to accept sparse data, by default None

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
        Random state, by default None
    accept_sparse : Union[bool, str], optional
        Whether to accept sparse data, by default None

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


def _pca_data(*views: np.ndarray):
    """
    Performs PCA on the data and returns the scores and loadings

    Parameters
    ----------
    views : np.ndarray

    Returns
    -------
    Us : list of np.ndarray
        The loadings for each view
    Ss : list of np.ndarray
        The scores for each view
    Vs : list of np.ndarray
        The eigenvectors for each view

    """
    principal_components = []
    for view in views:
        principal_components.append(PCA().fit(view))
    return principal_components
