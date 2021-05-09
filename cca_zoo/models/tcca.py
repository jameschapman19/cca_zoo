from typing import Tuple, List, Union

import numpy as np
import tensorly as tl
from scipy.linalg import sqrtm
from sklearn.metrics.pairwise import pairwise_kernels
from tensorly.decomposition import parafac

from .cca_base import _CCA_Base


class TCCA(_CCA_Base):
    """
    Fits a Tensor CCA model. Tensor CCA maximises higher order correlations

    My own port from https://github.com/rciszek/mdr_tcca

    :param latent_dims:
    :param c: regularisation between 0 (CCA) and 1 (PLS)

    :Example:

    >>> from cca_zoo.models import TCCA
    >>> X1 = np.random.rand(10,5)
    >>> X2 = np.random.rand(10,5)
    >>> model = TCCA()
    >>> model.fit(X1,X2)
    """

    def __init__(self, latent_dims: int = 1, c: List[float] = None):
        """
        Constructor for TCCA

        :param latent_dims:
        :param c: regularisation between 0 (CCA) and 1 (PLS)
        """
        super().__init__(latent_dims)
        self.c = c

    def check_params(self):
        if self.c is None:
            self.c = [0] * self.n_views

    def fit(self, *views: Tuple[np.ndarray, ...], ):
        self.n_views = len(views)
        self.check_params()
        assert (len(self.c) == len(views)), 'c requires as many values as #views'
        train_views, covs_invsqrt = self.setup_tensor(*views)
        for i, el in enumerate(train_views):
            if i == 0:
                M = el
            else:
                for _ in range(len(M.shape) - 1):
                    el = np.expand_dims(el, 1)
                M = np.expand_dims(M, -1) @ el
        M = np.mean(M, 0)
        tl.set_backend('numpy')
        M_parafac = parafac(M, self.latent_dims, verbose=True)
        self.alphas = [cov_invsqrt @ fac for i, (view, cov_invsqrt, fac) in
                       enumerate(zip(train_views, covs_invsqrt, M_parafac.factors))]
        self.score_list = [view @ self.alphas[i] for i, view in enumerate(train_views)]
        self.weights_list = [weights / np.linalg.norm(score) for weights, score in
                             zip(self.alphas, self.score_list)]
        self.score_list = [view @ self.weights_list[i] for i, view in enumerate(train_views)]
        self.train_correlations = self.predict_corr(*views)
        return self

    def setup_tensor(self, *views: Tuple[np.ndarray, ...]):
        train_views = self.demean_data(*views)
        n = train_views[0].shape[0]
        covs = [(1 - self.c[i]) * view.T @ view + self.c[i] * np.eye(view.shape[1]) for i, view in
                enumerate(train_views)]
        covs_invsqrt = [np.linalg.inv(sqrtm(cov)) for cov in covs]
        train_views = [train_view @ cov_invsqrt for train_view, cov_invsqrt in zip(train_views, covs_invsqrt)]
        return train_views, covs_invsqrt


class KTCCA(TCCA):
    """
    Fits a Kernel Tensor CCA model. Tensor CCA maximises higher order correlations

    My own port from https://github.com/rciszek/mdr_tcca

    :param latent_dims:
    :param c: regularisation between 0 (CCA) and 1 (PLS)

    :Example:

    >>> from cca_zoo.models import KTCCA
    >>> X1 = np.random.rand(10,5)
    >>> X2 = np.random.rand(10,5)
    >>> model = KTCCA()
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

    def setup_tensor(self, *views: Tuple[np.ndarray, ...]):
        self.train_views = self.demean_data(*views)
        train_views = [self._get_kernel(i, view) for i, view in enumerate(self.train_views)]
        n = train_views[0].shape[0]
        covs = [(1 - self.c[i]) * kernel @ kernel.T + self.c[i] * kernel for i, kernel in enumerate(train_views)]
        smallest_eigs = [min(0, np.linalg.eigvalsh(cov).min()) - self.eps for cov in covs]
        covs = [cov - smallest_eig * np.eye(cov.shape[0]) for cov, smallest_eig in zip(covs, smallest_eigs)]
        self.covs_invsqrt = [np.linalg.inv(sqrtm(cov)).real for cov in covs]
        train_views = [train_view @ cov_invsqrt for train_view, cov_invsqrt in zip(train_views, self.covs_invsqrt)]
        return train_views, self.covs_invsqrt

    def transform(self, *views: Tuple[np.ndarray, ...], ):
        """
        Transforms data given a fit k=KCCA model

        :param views: numpy arrays with the same number of rows (samples) separated by commas
        :param kwargs: any additional keyword arguments required by the given model
        """
        Ktest = [self._get_kernel(i, train_view, Y=test_view - self.view_means[i]) for i, (train_view, test_view) in
                 enumerate(zip(self.train_views, views))]
        transformed_views = [test_kernel.T @ cov_invsqrt @ self.alphas[i] for i, (test_kernel, cov_invsqrt) in
                             enumerate(zip(Ktest, self.covs_invsqrt))]
        return transformed_views
