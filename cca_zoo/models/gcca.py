from typing import Iterable, Union

import numpy as np
from scipy.linalg import eigh
from sklearn.metrics import pairwise_kernels
from sklearn.utils.validation import check_is_fitted

from cca_zoo.models import rCCA
from cca_zoo.utils.check_values import _process_parameter, check_views


class GCCA(rCCA):
    r"""
    A class used to fit GCCA model. For more than 2 views, GCCA optimizes the sum of correlations with a shared auxiliary vector

    :Maths:

    .. math::

        w_{opt}=\underset{w}{\mathrm{argmax}}\{ \sum_iw_i^TX_i^TT  \}\\

        \text{subject to:}

        T^TT=1

    :Citation:

    Tenenhaus, Arthur, and Michel Tenenhaus. "Regularized generalized canonical correlation analysis." Psychometrika 76.2 (2011): 257.

    :Example:

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
            view_cov = view.T @ view / self.n
            view_cov = (1 - self.c[i]) * view_cov + self.c[i] * np.eye(
                view_cov.shape[0]
            )
            Q.append(view_weight * view @ np.linalg.inv(view_cov) @ view.T)
        Q = np.sum(Q, axis=0)
        Q = (
            np.diag(np.sqrt(np.sum(K, axis=0)))
            @ Q
            @ np.diag(np.sqrt(np.sum(K, axis=0)))
        )
        return views, Q, None

    def _solve_evp(self, views: Iterable[np.ndarray], C, D=None, **kwargs):
        p = C.shape[0]
        [eigvals, eigvecs] = eigh(C, subset_by_index=[p - self.latent_dims, p - 1])
        idx = np.argsort(eigvals, axis=0)[::-1][: self.latent_dims]
        eigvecs = eigvecs[:, idx].real
        self.weights = [
            np.linalg.pinv(view) @ eigvecs[:, : self.latent_dims] for view in views
        ]


class KGCCA(GCCA):
    r"""
    A class used to fit KGCCA model. For more than 2 views, KGCCA optimizes the sum of correlations with a shared auxiliary vector

    :Maths:

    .. math::

        w_{opt}=\underset{w}{\mathrm{argmax}}\{ \sum_i\alpha_i^TK_i^TT  \}\\

        \text{subject to:}

        T^TT=1

    :Citation:

    Tenenhaus, Arthur, Cathy Philippe, and Vincent Frouin. "Kernel generalized canonical correlation analysis." Computational Statistics & Data Analysis 90 (2015): 114-131.

    :Example:

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
        """
        Constructor for PLS

        :param latent_dims: number of latent dimensions to fit
        :param scale: normalize variance in each column before fitting
        :param centre: demean data by column before fitting (and before transforming out of sample
        :param copy_data: If True, X will be copied; else, it may be overwritten
        :param random_state: Pass for reproducible output across multiple function calls
        :param c: Iterable of regularisation parameters for each view (between 0:CCA and 1:PLS)
        :param eps: epsilon for stability
        :param kernel: Iterable of kernel mappings used internally. This parameter is directly passed to :class:`~sklearn.metrics.pairwise.pairwise_kernel`. If element of `kernel` is a string, it must be one of the metrics in `pairwise.PAIRWISE_KERNEL_FUNCTIONS`. Alternatively, if element of `kernel` is a callable function, it is called on each pair of instances (rows) and the resulting value recorded. The callable should take two rows from X as input and return the corresponding kernel value as a single number. This means that callables from :mod:`sklearn.metrics.pairwise` are not allowed, as they operate on matrices, not single samples. Use the string identifying the kernel instead.
        :param gamma: Iterable of gamma parameters for the RBF, laplacian, polynomial, exponential chi2 and sigmoid kernels. Interpretation of the default value is left to the kernel; see the documentation for sklearn.metrics.pairwise. Ignored by other kernels.
        :param degree: Iterable of degree parameters of the polynomial kernel. Ignored by other kernels.
        :param coef0: Iterable of zero coefficients for polynomial and sigmoid kernels. Ignored by other kernels.
        :param kernel_params: Iterable of additional parameters (keyword arguments) for kernel function passed as callable object.
        :param eps: epsilon value to ensure stability of smallest eigenvalues
        """
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
            view_cov = view.T @ view / self.n
            view_cov = (1 - self.c[i]) * view_cov + self.c[i] * np.eye(
                view_cov.shape[0]
            )
            smallest_eig = min(0, np.linalg.eigvalsh(view_cov).min()) - self.eps
            view_cov = view_cov - smallest_eig * np.eye(view_cov.shape[0])
            Q.append(view_weight * view @ np.linalg.inv(view_cov) @ view.T)
        Q = np.sum(Q, axis=0)
        Q = (
            np.diag(np.sqrt(np.sum(K, axis=0)))
            @ Q
            @ np.diag(np.sqrt(np.sum(K, axis=0)))
        )
        return kernels, Q, None

    def transform(self, views: np.ndarray, y=None, **kwargs):
        """
        Transforms data given a fit KGCCA model

        :param views: list/tuple of numpy arrays or array likes with the same number of rows (samples)
        :param kwargs: any additional keyword arguments required by the given model
        """
        check_is_fitted(self, attributes=["weights"])
        views = check_views(
            *views, copy=self.copy_data, accept_sparse=self.accept_sparse
        )
        views = self._centre_scale_transform(views)
        Ktest = [
            self._get_kernel(i, self.train_views[i], Y=view)
            for i, view in enumerate(views)
        ]
        transformed_views = [
            kernel.T @ self.weights[i] for i, kernel in enumerate(Ktest)
        ]
        return transformed_views
