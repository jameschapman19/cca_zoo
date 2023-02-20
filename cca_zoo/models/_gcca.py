from typing import Iterable, Union

import numpy as np
from sklearn.metrics import pairwise_kernels
from sklearn.utils.validation import check_is_fitted

from cca_zoo.models._rcca import rCCA
from cca_zoo.utils.check_values import _process_parameter, _check_views


class GCCA(rCCA):
    r"""
    A class used to fit GCCA model. For more than 2 views, GCCA optimizes the sum of correlations with a shared auxiliary vector

    .. math::

        w_{opt}=\underset{w}{\mathrm{argmax}}\{ \sum_iw_i^TX_i^TT  \}\\

        \text{subject to:}

        T^TT=1

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
        Regularization parameter, by default None
    view_weights : Iterable[float], optional
        Weights for each view, by default None


    References
    ----------
    Tenenhaus, Arthur, and Michel Tenenhaus. "Regularized generalized canonical correlation analysis." Psychometrika 76.2 (2011): 257.

    Examples
    --------
    >>> from cca_zoo.models import GCCA
    >>> import numpy as np
    >>> rng=np.random.RandomState(0)
    >>> X1 = rng.random((10,5))
    >>> X2 = rng.random((10,5))
    >>> X3 = rng.random((10,5))
    >>> model = GCCA()
    >>> model.fit((X1,X2,X3)).score((X1,X2,X3))
    array([0.97229856])
    """

    def __init__(
        self,
        latent_dims: int = 1,
        scale: bool = True,
        centre=True,
        copy_data=True,
        random_state=None,
        c: Union[Iterable[float], float] = None,
        view_weights: Iterable[float] = None,
    ):
        super().__init__(
            latent_dims=latent_dims,
            scale=scale,
            centre=centre,
            copy_data=copy_data,
            accept_sparse=["csc", "csr"],
            random_state=random_state,
        )
        self.c = c
        self.view_weights = view_weights

    def _check_params(self):
        self.c = _process_parameter("c", self.c, 0, self.n_views)
        self.view_weights = _process_parameter(
            "view_weights", self.view_weights, 1, self.n_views
        )

    def _setup_evp(self, views: Iterable[np.ndarray], K=None):
        if K is None:
            # just use identity when all rows are observed in all views.
            K = np.ones((len(views), views[0].shape[0]))
        Q = []
        for i, (view, view_weight) in enumerate(zip(views, self.view_weights)):
            view_cov = (1 - self.c[i]) * np.cov(view, rowvar=False) + self.c[
                i
            ] * np.eye(view.shape[1])
            Q.append(view_weight * view @ np.linalg.inv(view_cov) @ view.T)
        Q = np.sum(Q, axis=0)
        Q = (
            np.diag(np.sqrt(np.sum(K, axis=0)))
            @ Q
            @ np.diag(np.sqrt(np.sum(K, axis=0)))
        )
        return Q, None

    def _weights(self, eigvals, eigvecs, views):
        self.weights = [
            np.linalg.pinv(view) @ eigvecs[:, : self.latent_dims] for view in views
        ]

    def _more_tags(self):
        return {"multiview": True}


class KGCCA(GCCA):
    r"""
    A class used to fit KGCCA model. For more than 2 views, KGCCA optimizes the sum of correlations with a shared auxiliary vector

    .. math::

        w_{opt}=\underset{w}{\mathrm{argmax}}\{ \sum_i\alpha_i^TK_i^TT  \}\\

        \text{subject to:}

        T^TT=1

    References
    ----------
    Tenenhaus, Arthur, Cathy Philippe, and Vincent Frouin. "Kernel generalized canonical correlation analysis." Computational Statistics & Data Analysis 90 (2015): 114-131.

    Examples
    --------
    >>> from cca_zoo.models import KGCCA
    >>> import numpy as np
    >>> rng=np.random.RandomState(0)
    >>> X1 = rng.random((10,5))
    >>> X2 = rng.random((10,5))
    >>> X3 = rng.random((10,5))
    >>> model = KGCCA()
    >>> model.fit((X1,X2,X3)).score((X1,X2,X3))
    array([0.97019284])
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
        kernel: Iterable[Union[float, callable]] = None,
        gamma: Iterable[float] = None,
        degree: Iterable[float] = None,
        coef0: Iterable[float] = None,
        kernel_params: Iterable[dict] = None,
    ):
        super().__init__(
            latent_dims=latent_dims,
            scale=scale,
            centre=centre,
            copy_data=copy_data,
            random_state=random_state,
        )
        self.kernel_params = kernel_params
        self.gamma = gamma
        self.coef0 = coef0
        self.kernel = kernel
        self.degree = degree
        self.c = c
        self.eps = eps

    def _check_params(self):
        self.kernel = _process_parameter("kernel", self.kernel, "linear", self.n_views)
        self.gamma = _process_parameter("gamma", self.gamma, None, self.n_views)
        self.coef0 = _process_parameter("coef0", self.coef0, 1, self.n_views)
        self.degree = _process_parameter("degree", self.degree, 1, self.n_views)
        self.c = _process_parameter("c", self.c, 0, self.n_views)
        self.view_weights = _process_parameter(
            "view_weights", self.view_weights, 1, self.n_views
        )

    def _get_kernel(self, view, X, Y=None):
        if callable(self.kernel[view]):
            params = self.kernel_params[view] or {}
        else:
            params = {
                "gamma": self.gamma[view],
                "degree": self.degree[view],
                "coef0": self.coef0[view],
            }
        return pairwise_kernels(
            X, Y, metric=self.kernel[view], filter_params=True, **params
        )

    def _setup_evp(self, views: Iterable[np.ndarray], K=None):
        self.train_views = views
        kernels = [self._get_kernel(i, view) for i, view in enumerate(self.train_views)]
        if K is None:
            # just use identity when all rows are observed in all views.
            K = np.ones((len(views), views[0].shape[0]))
        Q = []
        for i, (view, view_weight) in enumerate(zip(kernels, self.view_weights)):
            view_cov = (1 - self.c[i]) * np.cov(view, rowvar=False) + self.c[
                i
            ] * np.eye(view.shape[1])
            smallest_eig = min(0, np.linalg.eigvalsh(view_cov).min()) - self.eps
            view_cov = view_cov - smallest_eig * np.eye(view_cov.shape[0])
            Q.append(view_weight * view @ np.linalg.inv(view_cov) @ view.T)
        Q = np.sum(Q, axis=0)
        Q = (
            np.diag(np.sqrt(np.sum(K, axis=0)))
            @ Q
            @ np.diag(np.sqrt(np.sum(K, axis=0)))
        )
        self.splits = np.cumsum([0] + [kernel.shape[1] for kernel in kernels])
        return Q, None

    def transform(self, views: np.ndarray, y=None, **kwargs):
        check_is_fitted(self, attributes=["weights"])
        _check_views(views)
        views = [self.scalers[i].transform(view) for i, view in enumerate(views)]
        Ktest = [
            self._get_kernel(i, self.train_views[i], Y=view)
            for i, view in enumerate(views)
        ]
        transformed_views = [
            kernel.T @ self.weights[i] for i, kernel in enumerate(Ktest)
        ]
        return transformed_views

    def _weights(self, eigvals, eigvecs, views):
        kernels = [self._get_kernel(i, view) for i, view in enumerate(self.train_views)]
        self.weights = [
            np.linalg.pinv(kernel) @ eigvecs[:, : self.latent_dims]
            for kernel in kernels
        ]
