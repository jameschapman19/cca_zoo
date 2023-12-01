from typing import Iterable, Union

import numpy as np
from scipy.linalg import block_diag
from sklearn.metrics import pairwise_kernels
from sklearn.utils.validation import check_is_fitted

from cca_zoo._utils._checks import _process_parameter
from cca_zoo.linear._gcca import GCCA
from cca_zoo.linear._mcca import MCCA
from cca_zoo.linear._tcca import TCCA


class KernelMixin:
    def _check_params(self):
        self.kernel = _process_parameter("kernel", self.kernel, "linear", self.n_views_)
        self.gamma = _process_parameter("gamma", self.gamma, None, self.n_views_)
        self.coef0 = _process_parameter("coef0", self.coef0, 1, self.n_views_)
        self.degree = _process_parameter("degree", self.degree, 1, self.n_views_)
        self.c = _process_parameter("c", self.c, 0, self.n_views_)
        self.kernel_params = _process_parameter(
            "kernel_params", self.kernel_params, {}, self.n_views_
        )

    def _process_data(self, views, K=None):
        self.train_views = views
        kernels = [
            pairwise_kernels(
                view,
                metric=self.kernel[i],
                gamma=self.gamma[i],
                degree=self.degree[i],
                coef0=self.coef0[i],
                filter_params=True,
                **self.kernel_params[i]
            )
            for i, view in enumerate(self.train_views)
        ]
        return kernels

    def transform(self, views: Iterable[np.ndarray], **kwargs):
        check_is_fitted(self, attributes=["alphas"])
        Ktest = [
            pairwise_kernels(
                self.train_views[i],
                Y=view,
                metric=self.kernel[i],
                gamma=self.gamma[i],
                degree=self.degree[i],
                coef0=self.coef0[i],
                filter_params=True,
                **self.kernel_params[i]
            )
            for i, view in enumerate(views)
        ]
        transformed_views = [
            kernel.T @ self.alphas[i] for i, kernel in enumerate(Ktest)
        ]
        return transformed_views

    @property
    def alphas(self):
        check_is_fitted(self)
        return self.weights_

    def _more_tags(self):
        # Indicate that this class is for multiview data
        return {"kernel": True}


class KCCA(KernelMixin, MCCA):
    r"""
    A class used to fit KCCA model. This model extends MCCA to nonlinear relationships by using kernel functions on each view.

    The objective function of KCCA is:

    .. math::

        \alpha_{opt}=\underset{\alpha}{\mathrm{argmax}}\{\sum_i\sum_{j\neq i} \alpha_i^TK_i^TK_j\alpha_j  \}\\

        \text{subject to:}

        c_i\alpha_i^TK_i\alpha_i + (1-c_i)\alpha_i^TK_i^TK_i\alpha_i=1

    where:math:`K_i` are the kernel matrices for each view and:math:`c_i` are the regularization parameters for each view.

    Parameters
    ----------
    latent_dimensions: int, optional
        Number of latent dimensions to use, by default 1
    copy_data: bool, optional
        Whether to copy the data, by default True
    random_state: int, optional
        Random seed for reproducibility, by default None
    c: Union[Iterable[float], float], optional
        Regularization parameter or list of parameters for each view, by default None. If None, it will be set to zero for each view.
    eps: float, optional
        Small value to add to the diagonal of the kernel matrices, by default 1e-3
    kernel: Iterable[Union[float, callable]], optional
        Kernel function or list of functions for each view, by default None. If None, it will use a linear kernel for each view.
    gamma: Iterable[float], optional
        Gamma parameter or list of parameters for the RBF kernel for each view, by default None. Ignored if kernel is not RBF.
    degree: Iterable[float], optional
        Degree parameter or list of parameters for the polynomial kernel for each view, by default None. Ignored if kernel is not polynomial.
    coef0: Iterable[float], optional
        Coef0 parameter or list of parameters for the polynomial or sigmoid kernel for each view, by default None. Ignored if kernel is not polynomial or sigmoid.
    kernel_params: Iterable[dict], optional
        Additional parameters or list of parameters for the kernel function for each view, by default None.

    Examples
    --------
    >>> from cca_zoo.nonparametric import KCCA
    >>> import numpy as np
    >>> rng=np.random.RandomState(0)
    >>> X1 = rng.random((10,5))
    >>> X2 = rng.random((10,5))
    >>> X3 = rng.random((10,5))
    >>> model = KCCA()
    >>> model.fit((X1,X2,X3)).score((X1,X2,X3))
    array([0.96893666])
    """

    def __init__(
        self,
        latent_dimensions: int = 1,
        copy_data=True,
        random_state=None,
        c: Union[Iterable[float], float] = None,
        eps=1e-3,
        kernel: Iterable[Union[str, float, callable]] = None,
        gamma: Iterable[float] = None,
        degree: Iterable[float] = None,
        coef0: Iterable[float] = None,
        kernel_params: Iterable[dict] = None,
    ):
        # Call the parent class constructor
        super().__init__(
            latent_dimensions=latent_dimensions,
            copy_data=copy_data,
            random_state=random_state,
            c=c,
            eps=eps,
            pca=False,
        )
        # Store the kernel parameters
        self.kernel_params = kernel_params
        self.gamma = gamma
        self.coef0 = coef0
        self.kernel = kernel
        self.degree = degree

    def _D(self, views, **kwargs):
        D = block_diag(
            *[
                (1 - self.c[i]) * np.cov(view, rowvar=False) + self.c[i] * view
                for i, view in enumerate(views)
            ]
        )
        D_smallest_eig = min(0, np.linalg.eigvalsh(D).min()) - self.eps
        D = D - D_smallest_eig * np.eye(D.shape[0])
        return D / len(views)


class KMCCA(KCCA):
    r"""
    A class used to fit KMCCA model. This model extends KCCA to nonlinear relationships by using kernel functions on each view.

    The objective function of KMCCA is:

    .. math::

        \alpha_{opt}=\underset{\alpha}{\mathrm{argmax}}\{\alpha_1^TK_1^TK_2\alpha_2  \}\\

        \text{subject to:}

        c_i\alpha_i^TK_i\alpha_i + (1-c_i)\alpha_i^TK_i^TK_i\alpha_i=1

    where:math:`K_i` are the kernel matrices for each view and:math:`c_i` are the regularization parameters for each view.

    References
    ----------
    Hardoon, David R., et al. "Canonical correlation analysis: An overview with application to learning methods." Neural computation 16.12 (2004): 2639-2664.
    """


class KGCCA(KernelMixin, GCCA):
    r"""
    A class used to fit KGCCA model. This model extends GCCA to nonlinear relationships by using kernel functions on each view.

    The objective function of KGCCA is:

    .. math::

    \alpha_{opt}=\underset{\alpha}{\mathrm{argmax}}\{ \sum_i\alpha_i^TK_i^TT  \}\\

    \text{subject to:}

    T^TT=1

    where:math:`K_i` are the kernel matrices for each view and:math:`T` is the auxiliary vector.

    Parameters
    ----------
    latent_dimensions: int, optional
        Number of latent dimensions to use, by default 1
    copy_data: bool, optional
        Whether to copy the data, by default True
    random_state: int, optional
        Random seed for reproducibility, by default None
    c: Union[Iterable[float], float], optional
        Regularization parameter or list of parameters for each view, by default None. If None, it will be set to zero for each view.
    kernel: Iterable[Union[float, callable]], optional
        Kernel function or list of functions for each view, by default None. If None, it will use a linear kernel for each view.
    gamma: Iterable[float], optional
        Gamma parameter or list of parameters for the RBF kernel for each view, by default None. Ignored if kernel is not RBF.
    degree: Iterable[float], optional
        Degree parameter or list of parameters for the polynomial kernel for each view, by default None. Ignored if kernel is not polynomial.
    coef0: Iterable[float], optional
        Coef0 parameter or list of parameters for the polynomial or sigmoid kernel for each view, by default None. Ignored if kernel is not polynomial or sigmoid.
    kernel_params: Iterable[dict], optional
        Additional parameters or list of parameters for the kernel function for each view, by default None.
    view_weights: Iterable[float], optional
        Weights for each view in the objective function, by default None. If None, it will use equal weights for each view.

    References
    ----------
    Tenenhaus, Arthur, Cathy Philippe, and Vincent Frouin. "Kernel generalized canonical correlation analysis." Computational Statistics & Data Analysis 90 (2015): 114-131.

    Examples
    --------
    >>> from cca_zoo.nonparametric import KGCCA
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
        latent_dimensions: int = 1,
        copy_data=True,
        random_state=None,
        c: Union[Iterable[float], float] = None,
        kernel: Iterable[Union[float, callable]] = None,
        gamma: Iterable[float] = None,
        degree: Iterable[float] = None,
        coef0: Iterable[float] = None,
        kernel_params: Iterable[dict] = None,
        view_weights: Iterable[float] = None,
        eps: float = 1e-6,
    ):
        super().__init__(
            latent_dimensions=latent_dimensions,
            copy_data=copy_data,
            random_state=random_state,
            c=c,
            view_weights=view_weights,
            eps=eps,
        )
        self.kernel_params = kernel_params
        self.gamma = gamma
        self.coef0 = coef0
        self.kernel = kernel
        self.degree = degree

    def _weights(self, eigvals, eigvecs, views, **kwargs):
        kernels = [
            pairwise_kernels(
                view,
                metric=self.kernel[i],
                gamma=self.gamma[i],
                degree=self.degree[i],
                coef0=self.coef0[i],
                filter_params=True,
                **self.kernel_params[i]
            )
            for i, view in enumerate(self.train_views)
        ]
        self.weights_ = [
            np.linalg.pinv(kernel) @ eigvecs[:, : self.latent_dimensions]
            for kernel in kernels
        ]


class KTCCA(KernelMixin, TCCA):
    r"""
    A class used to fit KTCCA model. This model extends TCCA to nonlinear relationships by using kernel functions on each view.

    The objective function of KTCCA is:

    .. math::

        \alpha_{opt}=\underset{\alpha}{\mathrm{argmax}}\{ \alpha_1^TK_1^T\otimes \alpha_2^TK_2^T\otimes \cdots \otimes \alpha_m^TK_m^T\alpha  \}\\

        \text{subject to:}

        c_i\alpha_i^TK_i\alpha_i + (1-c_i)\alpha_i^TK_i^TK_i\alpha_i=1

    where:math:`K_i` are the kernel matrices for each view and:math:`c_i` are the regularization parameters for each view.

    References
    ----------
    Kim, Tae-Kyun, Shu-Fai Wong, and Roberto Cipolla. "Tensor canonical correlation analysis for action classification." 2007 IEEE Conference on Computer Vision and Pattern Recognition. IEEE, 2007

    Examples
    --------
    >>> from cca_zoo.nonparametric import KTCCA
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
        latent_dimensions: int = 1,
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
            latent_dimensions=latent_dimensions,
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

    def _setup_tensor(self, views: Iterable[np.ndarray], **kwargs):
        self.train_views = views
        kernels = [
            pairwise_kernels(
                view,
                metric=self.kernel[i],
                gamma=self.gamma[i],
                degree=self.degree[i],
                coef0=self.coef0[i],
                filter_params=True,
                **self.kernel_params[i]
            )
            for i, view in enumerate(self.train_views)
        ]
        return super()._setup_tensor(kernels)
