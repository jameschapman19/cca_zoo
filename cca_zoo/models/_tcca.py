from typing import Iterable, Union

import numpy as np
import tensorly as tl
from scipy.linalg import sqrtm
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils.validation import check_is_fitted
from tensorly.decomposition import parafac

from cca_zoo.models import MCCA
from cca_zoo.utils.check_values import _process_parameter


class TCCA(MCCA):
    r"""
    A class used to fit TCCA model. This model extends MCCA to higher order correlations by using tensor products of the views.

    The objective function of TCCA is:

    .. math::

        w_{opt}=\underset{w}{\mathrm{argmax}}\{ w_1^TX_1^T\otimes w_2^TX_2^T\otimes \cdots \otimes w_m^TX_m^Tw  \}\\

        \text{subject to:}

        w_i^TX_i^TX_iw_i=1

    where :math:`\otimes` denotes the Kronecker product.

    References
    ----------
    Kim, Tae-Kyun, Shu-Fai Wong, and Roberto Cipolla. "Tensor canonical correlation analysis for action classification." 2007 IEEE Conference on Computer Vision and Pattern Recognition. IEEE, 2007

    Examples
    --------
    >>> from cca_zoo.models import TCCA
    >>> rng=np.random.RandomState(0)
    >>> X1 = rng.random((10,5))
    >>> X2 = rng.random((10,5))
    >>> X3 = rng.random((10,5))
    >>> model = TCCA()
    >>> model.fit((X1,X2,X3)).score((X1,X2,X3))
    array([1.14595755])
    """

    def fit(self, views: Iterable[np.ndarray], y=None, **kwargs):
        self._validate_data(views)
        self._check_params()
        # returns whitened views along with whitening matrices
        whitened_views, covs_invsqrt = self._setup_tensor(*views)
        # The idea here is to form a matrix with M dimensions one for each view where at index
        # M[p_i,p_j,p_k...] we have the sum over n samples of the product of the pth feature of the
        # ith, jth, kth view etc.
        for i, el in enumerate(whitened_views):
            # To achieve this we start with the first view so M is nxp.
            if i == 0:
                M = el
            # For the remaining views we expand their dimensions to match M i.e. nx1x...x1xp
            else:
                for _ in range(len(M.shape) - 1):
                    el = np.expand_dims(el, 1)
                # Then we perform an outer product by expanding the dimensionality of M and
                # outer product with the expanded el
                M = np.expand_dims(M, -1) @ el
        M = np.mean(M, 0)
        tl.set_backend("numpy")
        M_parafac = parafac(M, self.latent_dims, verbose=False)
        self.weights = [
            cov_invsqrt @ fac
            for i, (view, cov_invsqrt, fac) in enumerate(
                zip(whitened_views, covs_invsqrt, M_parafac.factors)
            )
        ]
        return self

    def correlations(self, views: Iterable[np.ndarray], **kwargs):
        """
        Predicts the correlation for the given data using the fit model

        :param views: list/tuple of numpy arrays or array likes with the same number of rows (samples)
        :param kwargs: any additional keyword arguments required by the given model
        """
        transformed_views = self.transform(views, **kwargs)
        transformed_views = [
            transformed_view - transformed_view.mean(axis=0)
            for transformed_view in transformed_views
        ]
        multiplied_views = np.stack(transformed_views, axis=0).prod(axis=0).sum(axis=0)
        norms = np.stack(
            [
                np.linalg.norm(transformed_view, axis=0)
                for transformed_view in transformed_views
            ],
            axis=0,
        ).prod(axis=0)
        corrs = multiplied_views / norms
        return corrs

    def score(self, views: Iterable[np.ndarray], **kwargs):
        """
        Returns the higher order correlations in each dimension

        :param views: list/tuple of numpy arrays or array likes with the same number of rows (samples)
        :param kwargs: any additional keyword arguments required by the given model
        """
        dim_corrs = self.correlations(views, **kwargs)
        return dim_corrs

    def _setup_tensor(self, *views: np.ndarray, **kwargs):
        covs = [
            (1 - self.c[i]) * np.cov(view, rowvar=False)
            + self.c[i] * np.eye(view.shape[1])
            for i, view in enumerate(views)
        ]
        covs_invsqrt = [np.linalg.inv(sqrtm(cov)) for cov in covs]
        views = [
            train_view @ cov_invsqrt
            for train_view, cov_invsqrt in zip(views, covs_invsqrt)
        ]
        return views, covs_invsqrt

    def _more_tags(self):
        return {"multiview": True}


class KTCCA(TCCA):
    r"""
    A class used to fit KTCCA model. This model extends TCCA to nonlinear relationships by using kernel functions on each view.

    The objective function of KTCCA is:

    .. math::

        \alpha_{opt}=\underset{\alpha}{\mathrm{argmax}}\{ \alpha_1^TK_1^T\otimes \alpha_2^TK_2^T\otimes \cdots \otimes \alpha_m^TK_m^T\alpha  \}\\

        \text{subject to:}

        c_i\alpha_i^TK_i\alpha_i + (1-c_i)\alpha_i^TK_i^TK_i\alpha_i=1

    where :math:`K_i` are the kernel matrices for each view and :math:`c_i` are the regularization parameters for each view.

    References
    ----------
    Kim, Tae-Kyun, Shu-Fai Wong, and Roberto Cipolla. "Tensor canonical correlation analysis for action classification." 2007 IEEE Conference on Computer Vision and Pattern Recognition. IEEE, 2007

    Examples
    --------
    >>> from cca_zoo.models import KTCCA
    >>> rng=np.random.RandomState(0)
    >>> X1 = rng.random((10,5))
    >>> X2 = rng.random((10,5))
    >>> X3 = rng.random((10,5))
    >>> model = KTCCA()
    >>> model.fit((X1,X2,X3)).score((X1,X2,X3))
    array([1.69896269])
    """

    def __init__(
        self,
        latent_dims: int = 1,
        copy_data=True,
        random_state=None,
        eps=1e-3,
        c: Union[Iterable[float], float] = None,
        kernel: Iterable[Union[float, callable]] = None,
        gamma: Iterable[float] = None,
        degree: Iterable[float] = None,
        coef0: Iterable[float] = None,
        kernel_params: Iterable[dict] = None,
    ):
        super().__init__(
            latent_dims=latent_dims,
            copy_data=copy_data,
            random_state=random_state,
            eps=eps,
            c=c,
        )
        self.kernel_params = kernel_params
        self.gamma = gamma
        self.coef0 = coef0
        self.kernel = kernel
        self.degree = degree

    def _check_params(self):
        self.kernel = _process_parameter("kernel", self.kernel, "linear", self.n_views_)
        self.gamma = _process_parameter("gamma", self.gamma, None, self.n_views_)
        self.coef0 = _process_parameter("coef0", self.coef0, 1, self.n_views_)
        self.degree = _process_parameter("degree", self.degree, 1, self.n_views_)
        self.c = _process_parameter("c", self.c, 0, self.n_views_)

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

    def _setup_tensor(self, *views: np.ndarray):
        self.train_views = views
        kernels = [self._get_kernel(i, view) for i, view in enumerate(self.train_views)]
        covs = [
            (1 - self.c[i]) * np.cov(kernel, rowvar=False) + self.c[i] * kernel
            for i, kernel in enumerate(kernels)
        ]
        smallest_eigs = [
            min(0, np.linalg.eigvalsh(cov).min()) - self.eps for cov in covs
        ]
        covs = [
            cov - smallest_eig * np.eye(cov.shape[0])
            for cov, smallest_eig in zip(covs, smallest_eigs)
        ]
        covs_invsqrt = [np.linalg.inv(sqrtm(cov)).real for cov in covs]
        kernels = [
            kernel @ cov_invsqrt for kernel, cov_invsqrt in zip(kernels, covs_invsqrt)
        ]
        return kernels, covs_invsqrt

    def transform(self, views: np.ndarray, **kwargs):
        check_is_fitted(self, attributes=["weights"])
        Ktest = [
            self._get_kernel(i, self.train_views[i], Y=view)
            for i, view in enumerate(views)
        ]
        transformed_views = [
            kernel.T @ self.weights[i] for i, kernel in enumerate(Ktest)
        ]
        return transformed_views
