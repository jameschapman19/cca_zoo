from typing import Iterable, Union

import numpy as np
from scipy.linalg import block_diag
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils.validation import check_is_fitted

from cca_zoo.models._rcca import rCCA
from cca_zoo.utils.check_values import _process_parameter


class MCCA(rCCA):
    r"""
    A class used to fit MCCA model. This model extends CCA to more than two views by optimizing the sum of pairwise correlations.

    The objective function of MCCA is:

    .. math::

        w_{opt}=\underset{w}{\mathrm{argmax}}\{\sum_i\sum_{j\neq i} w_i^TX_i^TX_jw_j  \}\\

        \text{subject to:}

        (1-c_i)w_i^TX_i^TX_iw_i+c_iw_i^Tw_i=1

    where :math:`c_i` are the regularization parameters for each view.

    Parameters
    ----------
    latent_dims : int, optional
        Number of latent dimensions to use, by default 1
    copy_data : bool, optional
        Whether to copy the data, by default True
    random_state : int, optional
        Random seed for reproducibility, by default None
    c : Union[Iterable[float], float], optional
        Regularization parameter or list of parameters for each view, by default None. If None, it will be set to zero for each view.
    eps : float, optional
        Small value to add to the diagonal of the regularization matrix, by default 1e-9


    References
    ----------
    Kettenring, Jon R. "Canonical analysis of several sets of variables." Biometrika 58.3 (1971): 433-451.

    Examples
    --------
    >>> from cca_zoo.models import MCCA
    >>> import numpy as np
    >>> rng=np.random.RandomState(0)
    >>> X1 = rng.random((10,5))
    >>> X2 = rng.random((10,5))
    >>> X3 = rng.random((10,5))
    >>> model = MCCA()
    >>> model.fit((X1,X2,X3)).score((X1,X2,X3))
    array([0.97200847])
    """

    def __init__(
        self,
        latent_dims: int = 1,
        copy_data=True,
        random_state=None,
        c: Union[Iterable[float], float] = None,
        eps=1e-9,
    ):
        # Call the parent class constructor
        super().__init__(
            latent_dims=latent_dims,
            copy_data=copy_data,
            c=c,
            accept_sparse=["csc", "csr"],
            random_state=random_state,
        )
        self.eps = eps

    def _weights(self, eigvals, eigvecs, views):
        self.weights = [
            eigvecs[split : self.splits[i + 1]]
            for i, split in enumerate(self.splits[:-1])
        ]

    def _setup_evp(self, views: Iterable[np.ndarray], **kwargs):
        all_views = np.hstack(views)
        C = np.cov(all_views, rowvar=False)
        # Can regularise by adding to diagonal
        D = block_diag(
            *[
                (1 - self.c[i]) * np.cov(view, rowvar=False)
                + self.c[i] * np.eye(view.shape[1])
                for i, view in enumerate(views)
            ]
        )
        C -= block_diag(*[np.cov(view, rowvar=False) for view in views])
        D_smallest_eig = min(0, np.linalg.eigvalsh(D).min()) - self.eps
        D = D - D_smallest_eig * np.eye(D.shape[0])
        self.splits = np.cumsum([0] + self.n_features_)
        return C / self.n_views_, D / self.n_views_

    def _more_tags(self):
        return {"multiview": True}


class KCCA(MCCA):
    r"""
    A class used to fit KCCA model. This model extends MCCA to nonlinear relationships by using kernel functions on each view.

    The objective function of KCCA is:

    .. math::

        \alpha_{opt}=\underset{\alpha}{\mathrm{argmax}}\{\sum_i\sum_{j\neq i} \alpha_i^TK_i^TK_j\alpha_j  \}\\

        \text{subject to:}

        c_i\alpha_i^TK_i\alpha_i + (1-c_i)\alpha_i^TK_i^TK_i\alpha_i=1

    where :math:`K_i` are the kernel matrices for each view and :math:`c_i` are the regularization parameters for each view.

    Parameters
    ----------
    latent_dims : int, optional
        Number of latent dimensions to use, by default 1
    copy_data : bool, optional
        Whether to copy the data, by default True
    random_state : int, optional
        Random seed for reproducibility, by default None
    c : Union[Iterable[float], float], optional
        Regularization parameter or list of parameters for each view, by default None. If None, it will be set to zero for each view.
    eps : float, optional
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
    >>> from cca_zoo.models import KCCA
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
        latent_dims: int = 1,
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
        # Call the parent class constructor
        super().__init__(
            latent_dims=latent_dims,
            copy_data=copy_data,
            random_state=random_state,
            c=c,
            eps=eps,
        )
        # Store the kernel parameters
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

    def _setup_evp(self, views: Iterable[np.ndarray], **kwargs):
        self.train_views = views
        kernels = [self._get_kernel(i, view) for i, view in enumerate(self.train_views)]
        C = np.cov(np.hstack(kernels), rowvar=False)
        # Can regularise by adding to diagonal
        D = block_diag(
            *[
                (1 - self.c[i]) * np.cov(kernel, rowvar=False) + self.c[i] * kernel
                for i, kernel in enumerate(kernels)
            ]
        )
        C -= block_diag(*[np.cov(kernel, rowvar=False) for kernel in kernels]) - D
        D_smallest_eig = min(0, np.linalg.eigvalsh(D).min()) - self.eps
        D = D - D_smallest_eig * np.eye(D.shape[0])
        self.splits = np.cumsum([0] + [kernel.shape[1] for kernel in kernels])
        return C / self.n_views_, D / self.n_views_

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
