from typing import List, Union

import numpy as np
from scipy.linalg import block_diag, eigh
from sklearn.metrics.pairwise import pairwise_kernels

from .cca_base import _CCA_Base


# from hyperopt import fmin, tpe, Trials
class MCCA(_CCA_Base):
    """
    A class used to fit MCCA model. For more than 2 views, MCCA optimizes the sum of pairwise correlations.

    :Example:

    >>> from cca_zoo.models import MCCA
    >>> X1 = np.random.rand(10,5)
    >>> X2 = np.random.rand(10,5)
    >>> model = MCCA()
    >>> model.fit(X1,X2)
    """

    def __init__(self, latent_dims: int = 1, c: List[float] = None):
        """
        Constructor for MCCA

        :param latent_dims: number of latent dimensions
        :param c: list of regularisation parameters for each view (between 0:CCA and 1:PLS)
        """
        super().__init__(latent_dims=latent_dims)
        self.c = c

    def check_params(self):
        if self.c is None:
            self.c = [0] * self.n_views

    def fit(self, *views: np.ndarray):
        """
        Fits an MCCA model

        :param views: numpy arrays with the same number of rows (samples) separated by commas
        """
        self.n_views = len(views)
        self.check_params()
        assert (len(self.c) == len(views)), 'c requires as many values as #views'
        train_views, C, D = self.setup_gevp(*views)
        self.alphas = self.solve_gevp(C, D)
        self.score_list = [train_view @ eigvecs_ for train_view, eigvecs_ in zip(train_views, self.alphas)]
        self.weights_list = [weights / np.linalg.norm(score) for weights, score in zip(self.alphas, self.score_list)]
        self.score_list = [train_view @ weights for train_view, weights in zip(train_views, self.weights_list)]
        self.train_correlations = self.predict_corr(*views)
        return self

    def setup_gevp(self, *views: np.ndarray):
        train_views = self.centre_scale(*views)
        all_views = np.concatenate(train_views, axis=1)
        C = all_views.T @ all_views
        # Can regularise by adding to diagonal
        D = block_diag(*[(1 - self.c[i]) * m.T @ m + self.c[i] * np.eye(m.shape[1]) for i, m in enumerate(train_views)])
        C -= block_diag(*[view.T @ view for view in train_views]) - D
        self.splits = np.cumsum([0] + [view.shape[1] for view in train_views])
        return train_views, C, D

    def solve_gevp(self, C, D):
        n = D.shape[0]
        [eigvals, eigvecs] = eigh(C, D, subset_by_index=[n - self.latent_dims, n - 1])
        # sorting according to eigenvalue
        idx = np.argsort(eigvals, axis=0)[::-1][:self.latent_dims]
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

    def __init__(self, latent_dims: int = 1, c: List[float] = None, kernel: List[Union[float, callable]] = None,
                 gamma: List[float] = None,
                 degree: List[float] = None, coef0: List[float] = None,
                 kernel_params: List[dict] = None, eps=1e-3):
        """
        :param latent_dims: number of latent dimensions
        :param c: list of regularisation parameters for each view (between 0:CCA and 1:PLS)
        :param kernel: list of kernel mappings used internally. This parameter is directly passed to :class:`~sklearn.metrics.pairwise.pairwise_kernel`. If element of `kernel` is a string, it must be one of the metrics in `pairwise.PAIRWISE_KERNEL_FUNCTIONS`. Alternatively, if element of `kernel` is a callable function, it is called on each pair of instances (rows) and the resulting value recorded. The callable should take two rows from X as input and return the corresponding kernel value as a single number. This means that callables from :mod:`sklearn.metrics.pairwise` are not allowed, as they operate on matrices, not single samples. Use the string identifying the kernel instead.
        :param gamma: list of gamma parameters for the RBF, laplacian, polynomial, exponential chi2 and sigmoid kernels. Interpretation of the default value is left to the kernel; see the documentation for sklearn.metrics.pairwise. Ignored by other kernels.
        :param degree: list of degree parameters of the polynomial kernel. Ignored by other kernels.
        :param coef0: list of zero coefficients for polynomial and sigmoid kernels. Ignored by other kernels.
        :param kernel_params: list of additional parameters (keyword arguments) for kernel function passed as callable object.
        :param eps: epsilon value to ensure stability
        """
        super().__init__(latent_dims=latent_dims)
        self.kernel_params = kernel_params
        self.gamma = gamma
        self.coef0 = coef0
        self.kernel = kernel
        self.degree = degree
        self.c = c
        self.eps = eps

    def check_params(self):
        if self.kernel is None:
            self.kernel = ['linear'] * self.n_views
        if self.gamma is None:
            self.gamma = [None] * self.n_views
        if self.coef0 is None:
            self.coef0 = [1] * self.n_views
        if self.degree is None:
            self.degree = [1] * self.n_views
        if self.c is None:
            self.c = [0] * self.n_views

    def _get_kernel(self, view, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params[view] or {}
        else:
            params = {"gamma": self.gamma[view],
                      "degree": self.degree[view],
                      "coef0": self.coef0[view]}
        return pairwise_kernels(X, Y, metric=self.kernel[view],
                                filter_params=True, **params)

    def setup_gevp(self, *views: np.ndarray):
        """
        Generates the left and right hand sides of the generalized eigenvalue problem

        :param views:
        """
        self.train_views = self.centre_scale(*views)
        kernels = [self._get_kernel(i, view) for i, view in enumerate(self.train_views)]
        C = np.hstack(kernels).T @ np.hstack(kernels)
        # Can regularise by adding to diagonal
        D = block_diag(
            *[(1 - self.c[i]) * kernel @ kernel.T + self.c[i] * kernel for i, kernel in enumerate(kernels)])
        C -= block_diag(*[k.T @ k for k in kernels]) - D
        D_smallest_eig = min(0, np.linalg.eigvalsh(D).min()) - self.eps
        D = D - D_smallest_eig * np.eye(D.shape[0])
        self.splits = np.cumsum([0] + [kernel.shape[1] for kernel in kernels])
        return kernels, C, D

    def transform(self, *views: np.ndarray, view_indices: List[int] = None, **kwargs):
        """
        Transforms data given a fit KCCA model

        :param views: numpy arrays with the same number of rows (samples) separated by commas
        :param kwargs: any additional keyword arguments required by the given model
        """
        if view_indices is None:
            view_indices = np.arange(len(views))
        Ktest = [self._get_kernel(view_index, self.train_views[view_index], Y=test_view - self.view_means[view_index])
                 for test_view, view_index in
                 zip(views, view_indices)]
        transformed_views = [test_kernel.T @ self.alphas[view_index] for test_kernel, view_index in
                             zip(Ktest, view_indices)]
        return transformed_views
