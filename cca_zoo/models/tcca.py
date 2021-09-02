from typing import Iterable, Union

import numpy as np
import tensorly as tl
from scipy.linalg import sqrtm
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils.validation import check_is_fitted
from tensorly.decomposition import parafac

from .cca_base import _CCA_Base
from ..utils.check_values import _process_parameter, check_views


class TCCA(_CCA_Base):
    """
    Fits a Tensor CCA model. Tensor CCA maximises higher order correlations

    Citation
    --------
    Kim, Tae-Kyun, Shu-Fai Wong, and Roberto Cipolla. "Tensor canonical correlation analysis for action classification." 2007 IEEE Conference on Computer Vision and Pattern Recognition. IEEE, 2007

    My own port from https://github.com/rciszek/mdr_tcca

    :Example:

    >>> from cca_zoo.models import TCCA
    >>> X1 = np.random.rand(10,5)
    >>> X2 = np.random.rand(10,5)
    >>> model = TCCA()
    >>> model.fit(X1,X2)
    """

    def __init__(self, latent_dims: int = 1, scale=True, centre=True, copy_data=True, random_state=None,
                 c: Union[Iterable[float], float] = None):
        """
        Constructor for TCCA

        :param latent_dims: number of latent dimensions to fit
        :param scale: normalize variance in each column before fitting
        :param centre: demean data by column before fitting (and before transforming out of sample
        :param copy_data: If True, X will be copied; else, it may be overwritten
        :param random_state: Pass for reproducible output across multiple function calls
        :param c: Iterable of regularisation parameters for each view (between 0:CCA and 1:PLS)
        """
        super().__init__(latent_dims=latent_dims, scale=scale, centre=centre, copy_data=copy_data,
                         accept_sparse=['csc', 'csr'],
                         random_state=random_state)
        self.c = c

    def _check_params(self):
        self.c = _process_parameter('c', self.c, 0, self.n_views)

    def fit(self, *views: np.ndarray, ):
        views = check_views(*views, copy=self.copy_data, accept_sparse=self.accept_sparse)
        views = self._centre_scale(*views)
        self.n_views = len(views)
        self.n = views[0].shape[0]
        self._check_params()
        # returns whitened views along with whitening matrix
        views, covs_invsqrt = self._setup_tensor(*views)
        for i, el in enumerate(views):
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
                       enumerate(zip(views, covs_invsqrt, M_parafac.factors))]
        self.weights = self.alphas
        return self

    def _setup_tensor(self, *views: np.ndarray, **kwargs):
        train_views = self._centre_scale(*views)
        n = train_views[0].shape[0]
        covs = [(1 - self.c[i]) * view.T @ view / (self.n) + self.c[i] * np.eye(view.shape[1]) for i, view in
                enumerate(train_views)]
        covs_invsqrt = [np.linalg.inv(sqrtm(cov)) for cov in covs]
        train_views = [train_view @ cov_invsqrt for train_view, cov_invsqrt in zip(train_views, covs_invsqrt)]
        return train_views, covs_invsqrt


class KTCCA(TCCA):
    """
    Fits a Kernel Tensor CCA model. Tensor CCA maximises higher order correlations

    Citation
    --------
    Kim, Tae-Kyun, Shu-Fai Wong, and Roberto Cipolla. "Tensor canonical correlation analysis for action classification." 2007 IEEE Conference on Computer Vision and Pattern Recognition. IEEE, 2007

    :Example:

    >>> from cca_zoo.models import KTCCA
    >>> X1 = np.random.rand(10,5)
    >>> X2 = np.random.rand(10,5)
    >>> model = KTCCA()
    >>> model.fit(X1,X2)
    """

    def __init__(self, latent_dims: int = 1, scale: bool = True, centre=True, copy_data=True, random_state=None,
                 eps=1e-3, c: Union[Iterable[float], float] = None,
                 kernel: Iterable[Union[float, callable]] = None,
                 gamma: Iterable[float] = None,
                 degree: Iterable[float] = None, coef0: Iterable[float] = None,
                 kernel_params: Iterable[dict] = None):
        """
        Constructor for TCCA

        :param latent_dims: number of latent dimensions to fit
        :param scale: normalize variance in each column before fitting
        :param centre: demean data by column before fitting (and before transforming out of sample
        :param copy_data: If True, X will be copied; else, it may be overwritten
        :param random_state: Pass for reproducible output across multiple function calls
        :param c: Iterable of regularisation parameters for each view (between 0:CCA and 1:PLS)
        :param kernel: Iterable of kernel mappings used internally. This parameter is directly passed to :class:`~sklearn.metrics.pairwise.pairwise_kernel`. If element of `kernel` is a string, it must be one of the metrics in `pairwise.PAIRWISE_KERNEL_FUNCTIONS`. Alternatively, if element of `kernel` is a callable function, it is called on each pair of instances (rows) and the resulting value recorded. The callable should take two rows from X as input and return the corresponding kernel value as a single number. This means that callables from :mod:`sklearn.metrics.pairwise` are not allowed, as they operate on matrices, not single samples. Use the string identifying the kernel instead.
        :param gamma: Iterable of gamma parameters for the RBF, laplacian, polynomial, exponential chi2 and sigmoid kernels. Interpretation of the default value is left to the kernel; see the documentation for sklearn.metrics.pairwise. Ignored by other kernels.
        :param degree: Iterable of degree parameters of the polynomial kernel. Ignored by other kernels.
        :param coef0: Iterable of zero coefficients for polynomial and sigmoid kernels. Ignored by other kernels.
        :param kernel_params: Iterable of additional parameters (keyword arguments) for kernel function passed as callable object.
        :param eps: epsilon value to ensure stability
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

    def _setup_tensor(self, *views: np.ndarray):
        self.train_views = views
        kernels = [self._get_kernel(i, view) for i, view in enumerate(self.train_views)]
        covs = [(1 - self.c[i]) * kernel @ kernel.T / (self.n - 1) + self.c[i] * kernel for i, kernel in
                enumerate(kernels)]
        smallest_eigs = [min(0, np.linalg.eigvalsh(cov).min()) - self.eps for cov in covs]
        covs = [cov - smallest_eig * np.eye(cov.shape[0]) for cov, smallest_eig in zip(covs, smallest_eigs)]
        self.covs_invsqrt = [np.linalg.inv(sqrtm(cov)).real for cov in covs]
        kernels = [kernel @ cov_invsqrt for kernel, cov_invsqrt in zip(kernels, self.covs_invsqrt)]
        return kernels, self.covs_invsqrt

    def transform(self, *views: np.ndarray, view_indices: Iterable[int] = None, **kwargs):
        """
        Transforms data given a fit k=KCCA model

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
                 for test_view, view_index in
                 zip(transformed_views, view_indices)]
        transformed_views = [test_kernel.T @ cov_invsqrt @ self.alphas[view_index] for
                             i, (test_kernel, view_index, cov_invsqrt) in
                             enumerate(zip(Ktest, view_indices, self.covs_invsqrt))]
        return transformed_views
