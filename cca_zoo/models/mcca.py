from typing import Iterable, Union

import numpy as np
from scipy.linalg import block_diag, eigh
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils.validation import check_is_fitted

from .cca_base import _CCA_Base
from ..utils.check_values import _process_parameter, check_views


class MCCA(_CCA_Base):
    """
    A class used to fit MCCA model. For more than 2 views, MCCA optimizes the sum of pairwise correlations.

    Citation
    --------
    Kettenring, Jon R. "Canonical analysis of several sets of variables." Biometrika 58.3 (1971): 433-451.

    :Example:
    >>> from cca_zoo.models import MCCA
    >>> X1 = np.random.rand(10,5)
    >>> X2 = np.random.rand(10,5)
    >>> model = MCCA()
    >>> model.fit(X1,X2)
    """

    def __init__(self, latent_dims: int = 1, scale: bool = True, centre=True, copy_data=True, random_state=None,
                 c: Union[Iterable[float], float] = None,
                 eps=1e-3):
        """
        Constructor for MCCA

        :param latent_dims: number of latent dimensions to fit
        :param scale: normalize variance in each column before fitting
        :param centre: demean data by column before fitting (and before transforming out of sample
        :param copy_data: If True, X will be copied; else, it may be overwritten
        :param random_state: Pass for reproducible output across multiple function calls
        :param c: Iterable of regularisation parameters for each view (between 0:CCA and 1:PLS)
        :param eps: epsilon for stability
        """
        super().__init__(latent_dims=latent_dims, scale=scale, centre=centre, copy_data=copy_data,
                         accept_sparse=['csc', 'csr'],
                         random_state=random_state)
        self.c = c
        self.eps = eps

    def _check_params(self):
        self.c = _process_parameter('c', self.c, 0, self.n_views)

    def fit(self, *views: np.ndarray):
        """
        Fits an MCCA model

        :param views: numpy arrays with the same number of rows (samples) separated by commas
        """
        views = check_views(*views, copy=self.copy_data, accept_sparse=self.accept_sparse)
        views = self._centre_scale(*views)
        self.n_views = len(views)
        self.n = views[0].shape[0]
        self._check_params()
        views, C, D = self._setup_gevp(*views)
        self.alphas = self._solve_gevp(C, D)
        self.weights = self.alphas
        return self

    def _setup_gevp(self, *views: np.ndarray):
        all_views = np.concatenate(views, axis=1)
        C = all_views.T @ all_views / self.n
        # Can regularise by adding to diagonal
        D = block_diag(
            *[(1 - self.c[i]) * m.T @ m / self.n + self.c[i] * np.eye(m.shape[1]) for i, m in enumerate(views)])
        C -= block_diag(*[view.T @ view / self.n for view in views]) - D
        D_smallest_eig = min(0, np.linalg.eigvalsh(D).min()) - self.eps
        D = D - D_smallest_eig * np.eye(D.shape[0])
        self.splits = np.cumsum([0] + [view.shape[1] for view in views])
        return views, C, D

    def _solve_gevp(self, C, D):
        n = D.shape[0]
        [eigvals, eigvecs] = eigh(C, D, subset_by_index=[n - self.latent_dims, n - 1])
        # sorting according to eigenvalue
        idx = np.argsort(eigvals, axis=0)[::-1][:self.latent_dims]
        eigvecs = eigvecs * eigvals
        eigvecs = eigvecs[:, idx].real
        eigvecs = [eigvecs[split:self.splits[i + 1]] for i, split in enumerate(self.splits[:-1])]
        return eigvecs


class KCCA(MCCA):
    """
    A class used to fit KCCA model.

    :Example:

    >>> from cca_zoo.models import KCCA
    >>> X1 = np.random.rand(10,5)
    >>> X2 = np.random.rand(10,5)
    >>> model = KCCA()
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

    def _get_kernel(self, view, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params[view] or {}
        else:
            params = {"gamma": self.gamma[view],
                      "degree": self.degree[view],
                      "coef0": self.coef0[view]}
        return pairwise_kernels(X, Y, metric=self.kernel[view],
                                filter_params=True, **params)

    def _setup_gevp(self, *views: np.ndarray):
        """
        Generates the left and right hand sides of the generalized eigenvalue problem

        :param views:
        """
        self.train_views = views
        kernels = [self._get_kernel(i, view) for i, view in enumerate(self.train_views)]
        C = np.hstack(kernels).T @ np.hstack(kernels) / self.n
        # Can regularise by adding to diagonal
        D = block_diag(
            *[(1 - self.c[i]) * kernel @ kernel.T / self.n + self.c[i] * kernel for i, kernel in enumerate(kernels)])
        C -= block_diag(*[kernel.T @ kernel / self.n for kernel in kernels]) - D
        D_smallest_eig = min(0, np.linalg.eigvalsh(D).min()) - self.eps
        D = D - D_smallest_eig * np.eye(D.shape[0])
        self.splits = np.cumsum([0] + [kernel.shape[1] for kernel in kernels])
        return kernels, C, D

    def transform(self, *views: np.ndarray, view_indices: Iterable[int] = None, **kwargs):
        """
        Transforms data given a fit KCCA model

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
