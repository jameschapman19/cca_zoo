from typing import Iterable, Union

import numpy as np
from scipy.linalg import eigh
from sklearn.metrics import pairwise_kernels
from sklearn.utils.validation import check_is_fitted

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
                 c: Union[Iterable[float], float] = None,
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
        Q = self._setup_evp(*views, K=K)
        self._solve_evp(*views, Q=Q)
        return self

    def _setup_evp(self, *views: np.ndarray, K=None):
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
        return Q

    def _solve_evp(self, *views, Q=None):
        n = Q.shape[0]
        [eigvals, eigvecs] = eigh(Q, subset_by_index=[n - self.latent_dims, n - 1])
        idx = np.argsort(eigvals, axis=0)[::-1][:self.latent_dims]
        eigvecs = eigvecs[:, idx].real
        self.weights = [np.linalg.pinv(view) @ eigvecs[:, :self.latent_dims] for view in views]


class KGCCA(GCCA):
    """
    A class used to fit KGCCA model. For more than 2 views, KGCCA optimizes the sum of correlations with a shared auxiliary vector

    Citation
    --------
    Tenenhaus, Arthur, Cathy Philippe, and Vincent Frouin. "Kernel generalized canonical correlation analysis." Computational Statistics & Data Analysis 90 (2015): 114-131.

    :Example:

    >>> from cca_zoo.models import KGCCA
    >>> rng=np.random.RandomState(0)
    >>> X1 = rng.random(10,5)
    >>> X2 = np.random.rand(10,5)
    >>> model = KGCCA()
    >>> model.fit(X1,X2)
    """

    def __init__(self, latent_dims: int = 1, scale: bool = True, centre=True, copy_data=True, random_state=None,
                 c: Union[Iterable[float], float] = None, eps=1e-3,
                 kernel: Iterable[Union[float, callable]] = None,
                 gamma: Iterable[float] = None,
                 degree: Iterable[float] = None, coef0: Iterable[float] = None,
                 kernel_params: Iterable[dict] = None):
        """
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
        super().__init__(latent_dims=latent_dims, scale=scale, centre=centre, copy_data=copy_data,
                         random_state=random_state)
        self.kernel_params = kernel_params
        self.gamma = gamma
        self.coef0 = coef0
        self.kernel = kernel
        self.degree = degree
        self.c = c
        self.eps = eps

    def _check_params(self):
        self.kernel = _process_parameter('kernel', self.kernel, 'linear', self.n_views)
        self.gamma = _process_parameter('gamma', self.gamma, None, self.n_views)
        self.coef0 = _process_parameter('coef0', self.coef0, 1, self.n_views)
        self.degree = _process_parameter('degree', self.degree, 1, self.n_views)
        self.c = _process_parameter('c', self.c, 0, self.n_views)
        self.view_weights = _process_parameter('view_weights', self.view_weights, 1, self.n_views)

    def _get_kernel(self, view, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params[view] or {}
        else:
            params = {"gamma": self.gamma[view],
                      "degree": self.degree[view],
                      "coef0": self.coef0[view]}
        return pairwise_kernels(X, Y, metric=self.kernel[view],
                                filter_params=True, **params)

    def _setup_evp(self, *views: np.ndarray, K=None):
        self.train_views = views
        kernels = [self._get_kernel(i, view) for i, view in enumerate(self.train_views)]
        if K is None:
            # just use identity when all rows are observed in all views.
            K = np.ones((len(views), views[0].shape[0]))
        Q = []
        for i, (view, view_weight) in enumerate(zip(kernels, self.view_weights)):
            view_cov = view.T @ view / self.n
            view_cov = (1 - self.c[i]) * view_cov + self.c[i] * np.eye(view_cov.shape[0])
            smallest_eig = min(0, np.linalg.eigvalsh(view_cov).min()) - self.eps
            view_cov = view_cov - smallest_eig * np.eye(view_cov.shape[0])
            Q.append(view_weight * view @ np.linalg.inv(view_cov) @ view.T)
        Q = np.sum(Q, axis=0)
        Q = np.diag(np.sqrt(np.sum(K, axis=0))) @ Q @ np.diag(np.sqrt(np.sum(K, axis=0)))
        return Q

    def _solve_evp(self, *views, Q=None):
        kernels = [self._get_kernel(i, view) for i, view in enumerate(self.train_views)]
        n = Q.shape[0]
        [eigvals, eigvecs] = eigh(Q, subset_by_index=[n - self.latent_dims, n - 1])
        idx = np.argsort(eigvals, axis=0)[::-1][:self.latent_dims]
        eigvecs = eigvecs[:, idx].real
        self.alphas = [np.linalg.pinv(kernel) @ eigvecs[:, :self.latent_dims] for kernel in kernels]

    def transform(self, *views: np.ndarray, view_indices: Iterable[int] = None, **kwargs):
        """
        Transforms data given a fit KGCCA model

        :param views: numpy arrays with the same number of rows (samples) separated by commas
        :param kwargs: any additional keyword arguments required by the given model
        """
        check_is_fitted(self, attributes=['alphas'])
        views = check_views(*views, copy=self.copy_data, accept_sparse=self.accept_sparse)
        if view_indices is None:
            view_indices = np.arange(len(views))
        transformed_views = []
        for i, (view, view_index) in enumerate(zip(views, view_indices)):
            if self.centre:
                view = view - self.view_means[view_index]
            if self.scale:
                view = view / self.view_stds[view_index]
            transformed_views.append(view)
        Ktest = [self._get_kernel(view_index, self.train_views[view_index], Y=test_view)
                 for test_view, view_index in zip(transformed_views, view_indices)]
        transformed_views = [test_kernel.T @ self.alphas[view_index] for test_kernel, view_index in
                             zip(Ktest, view_indices)]
        return transformed_views
