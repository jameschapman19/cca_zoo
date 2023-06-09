from scipy.linalg import block_diag
from sklearn.decomposition import PCA
import numpy as np

from typing import Iterable, Union
from scipy.linalg import eigh
from cca_zoo.models._plsmixin import PLSMixin
from cca_zoo.models._base import BaseModel
from cca_zoo.utils import _process_parameter


class MCCA(BaseModel):
    r"""
    A class used to fit Regularised CCA (canonical ridge) model. This model adds a regularization term to the CCA objective function to avoid overfitting and improve stability. It uses PCA to perform the optimization efficiently for high dimensional data.

    The objective function of regularised CCA is:

    .. math::

        w_{opt}=\underset{w}{\mathrm{argmax}}\{ w_1^TX_1^TX_2w_2  \}\\

        \text{subject to:}

        (1-c_1)w_1^TX_1^TX_1w_1+c_1w_1^Tw_1=n

        (1-c_2)w_2^TX_2^TX_2w_2+c_2w_2^Tw_2=n

    where :math:`c_i` are the regularization parameters for each view.

    Parameters
    ----------
    latent_dims : int, optional
        Number of latent dimensions to use, by default 1
    copy_data : bool, optional
        Whether to copy the data, by default True
    random_state : int, optional
        Random state, by default None
    c : Union[Iterable[float], float], optional
        Regularisation parameter, by default None
    accept_sparse : Union[bool, str], optional
        Whether to accept sparse data, by default None


    References
    --------
    Vinod, Hrishikesh D. "Canonical ridge and econometrics of joint production." Journal of econometrics 4.2 (1976): 147-166.
    """

    def __init__(
        self,
        latent_dims: int = 1,
        copy_data=True,
        random_state=None,
        c: Union[Iterable[float], float] = None,
        accept_sparse=None,
        eps: float = 1e-6,
        pca: bool = True,
    ):
        # Set the default value for accept_sparse
        if accept_sparse is None:
            accept_sparse = ["csc", "csr"]
        # Call the parent class constructor
        super().__init__(
            latent_dims=latent_dims,
            copy_data=copy_data,
            accept_sparse=accept_sparse,
            random_state=random_state,
        )
        # Store the c parameter
        self.c = c
        self.eps = eps
        self.pca = pca

    def _check_params(self):
        # Process the c parameter for each view
        self.c = _process_parameter("c", self.c, 0, self.n_views_)

    def fit(self, views: Iterable[np.ndarray], y=None, **kwargs):
        # Validate the input data
        self._validate_data(views)
        # Check the parameters
        self._check_params()
        views = self._process_data(views, **kwargs)
        eigvals, eigvecs = self._solve_gevp(views, y=y, **kwargs)
        # Compute the weights for each view
        self._weights(eigvals, eigvecs, views, **kwargs)
        return self

    def _process_data(self, views, **kwargs):
        if self.pca:
            views = self._apply_pca(views)
        return views

    def _solve_gevp(self, views: Iterable[np.ndarray], y=None, **kwargs):
        # Setup the eigenvalue problem
        C = self.C(views, **kwargs)
        D = self.D(views, **kwargs)
        self.splits = np.cumsum([view.shape[1] for view in views])
        # Solve the eigenvalue problem
        # Get the dimension of C
        p = C.shape[0]
        # Solve the generalized eigenvalue problem Cx=lambda Dx using a subset of eigenvalues and eigenvectors
        [eigvals, eigvecs] = eigh(
            C,
            D,
            subset_by_index=[p - self.latent_dims, p - 1],
        )
        # Sort the eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigvals, axis=0)[::-1]
        eigvecs = eigvecs[:, idx].real
        return np.flip(eigvals), eigvecs

    def _weights(self, eigvals, eigvecs, views, **kwargs):
        # split eigvecs into weights for each view
        self.weights = np.split(eigvecs, self.splits[:-1], axis=0)
        if self.pca:
            # go from weights in PCA space to weights in original space
            self.weights = [
                pca.components_.T @ self.weights[i] for i, pca in enumerate(self.pca)
            ]

    def _apply_pca(self, views):
        """
        Do data driven PCA on each view
        """
        self.pca = [PCA() for _ in views]
        # Fit PCA on each view
        return [self.pca[i].fit_transform(view) for i, view in enumerate(views)]

    def C(self, views, **kwargs):
        all_views = np.hstack(views)
        C = np.cov(all_views, rowvar=False)
        C -= block_diag(*[np.cov(view, rowvar=False) for view in views])
        return C / len(views)

    def D(self, views, **kwargs):
        if self.pca:
            # Can regularise by adding to diagonal
            D = block_diag(
                *[
                    np.diag((1 - self.c[i]) * pc.explained_variance_ + self.c[i])
                    for i, pc in enumerate(self.pca)
                ]
            )
        else:
            D = block_diag(
                *[
                    (1 - self.c[i]) * np.cov(view, rowvar=False)
                    + self.c[i] * np.eye(view.shape[1])
                    for i, view in enumerate(views)
                ]
            )
        D_smallest_eig = min(0, np.linalg.eigvalsh(D).min()) - self.eps
        D = D - D_smallest_eig * np.eye(D.shape[0])
        return D / len(views)

    def _more_tags(self):
        # Indicate that this class is for multiview data
        return {"multiview": True}


class rCCA(MCCA):
    r"""
    A class used to fit Regularised CCA (canonical ridge) model. This model adds a regularization term to the CCA objective function to avoid overfitting and improve stability. It uses PCA to perform the optimization efficiently for high dimensional data.

    The objective function of regularised CCA is:

    .. math::

        w_{opt}=\underset{w}{\mathrm{argmax}}\{ w_1^TX_1^TX_2w_2  \}\\

        \text{subject to:}

        (1-c_1)w_1^TX_1^TX_1w_1+c_1w_1^Tw_1=n

        (1-c_2)w_2^TX_2^TX_2w_2+c_2w_2^Tw_2=n

    where :math:`c_i` are the regularization parameters for each view.

    References
    --------
    Vinod, Hrishikesh D. "Canonical ridge and econometrics of joint production." Journal of econometrics 4.2 (1976): 147-166.
    """

    def C(self, views, **kwargs):
        if len(views) != 2:
            raise ValueError(
                f"Model can only be used with two views, but {len(views)} were given. Use MCCA or GCCA instead."
            )
        # Compute the B matrices for each view
        B = [
            (1 - self.c[i]) * pc.explained_variance_ + self.c[i]
            for i, pc in enumerate(self.pca)
        ]
        C = np.cov(views[0] / np.sqrt(B[0]), views[1] / np.sqrt(B[1]), rowvar=False)[
            0 : views[0].shape[1], views[0].shape[1] :
        ]
        # if views[0].shape[1] <= views[1].shape[1] then return R@R^T else return R^T@R
        if views[0].shape[1] <= views[1].shape[1]:
            self.primary_view = 0
            return C @ C.T
        else:
            self.primary_view = 1
            return C.T @ C

    def D(self, views, **kwargs):
        return None

    def _weights(self, eigvals, eigvecs, views):
        B = [
            (1 - self.c[i]) * pc.singular_values_**2 / self.n_samples_ + self.c[i]
            for i, pc in enumerate(self.pca)
        ]
        C = np.cov(
            views[self.primary_view], views[1 - self.primary_view], rowvar=False
        )[0 : views[self.primary_view].shape[1], views[self.primary_view].shape[1] :]
        self.weights = [None] * 2
        # Compute the weight matrix for primary view
        self.weights[1 - self.primary_view] = (
            # Project view 1 onto its principal components
            self.pca[1 - self.primary_view].components_.T
            # Scale by the inverse of B[0]
            @ np.diag(1 / B[1 - self.primary_view])
            # Multiply by the cross-covariance matrix
            @ C.T
            # Scale by the inverse of the square root of B[1]
            @ np.diag(1 / np.sqrt(B[self.primary_view]))
            # Multiply by the eigenvectors
            @ eigvecs
            # Scale by the inverse of the square root of eigenvalues
            / np.sqrt(eigvals)
        )

        # Compute the weight matrix for view 2
        self.weights[self.primary_view] = (
            # Project view 2 onto its principal components
            self.pca[self.primary_view].components_.T
            # Scale by the inverse of the square root of B[1]
            @ np.diag(1 / np.sqrt(B[self.primary_view]))
            # Multiply by the eigenvectors
            @ eigvecs
        )


class CCA(rCCA):
    r"""
    A class used to fit a simple CCA model. This model finds the linear projections of two views that maximize their correlation.

    The objective function of CCA is:

    .. math::

        w_{opt}=\underset{w}{\mathrm{argmax}}\{ w_1^TX_1^TX_2w_2  \}\\

        \text{subject to:}

        w_1^TX_1^TX_1w_1=n

        w_2^TX_2^TX_2w_2=n

    Parameters
    ----------
    latent_dims : int, optional
        Number of latent dimensions to use, by default 1
    copy_data : bool, optional
        Whether to copy the data, by default True
    random_state : int, optional
        Random seed for reproducibility, by default None

    References
    --------

    Hotelling, Harold. "Relations between two sets of variates." Breakthroughs in statistics. Springer, New York, NY, 1992. 162-190.

    Example
    -------
    >>> from cca_zoo.models import CCA
    >>> import numpy as np
    >>> rng=np.random.RandomState(0)
    >>> X1 = rng.random((10,5))
    >>> X2 = rng.random((10,5))
    >>> model = CCA()
    >>> model.fit((X1,X2)).score((X1,X2))
    array([1.])
    """

    def __init__(
        self,
        latent_dims: int = 1,
        copy_data=True,
        random_state=None,
    ):
        # Call the parent class constructor with c=0.0 to disable regularization
        super().__init__(
            latent_dims=latent_dims,
            copy_data=copy_data,
            c=0.0,
            random_state=random_state,
        )


class PLS(MCCA, PLSMixin):
    r"""
    A class used to fit a simple PLS model. This model finds the linear projections of two views that maximize their covariance.

    Implements PLS by inheriting regularised CCA with maximal regularisation. This is equivalent to solving the following optimization problem:

    .. math::

        w_{opt}=\underset{w}{\mathrm{argmax}}\{ w_1^TX_1^TX_2w_2  \}\\

        \text{subject to:}

        w_1^Tw_1=1

        w_2^Tw_2=1

    Parameters
    ----------
    latent_dims : int, optional
        Number of latent dimensions to use, by default 1
    copy_data : bool, optional
        Whether to copy the data, by default True
    random_state : int, optional
        Random state, by default None
    """

    def __init__(
        self,
        latent_dims: int = 1,
        copy_data=True,
        random_state=None,
    ):
        # Call the parent class constructor with c=1 to enable maximal regularization
        super().__init__(
            latent_dims=latent_dims,
            copy_data=copy_data,
            c=1,
            random_state=random_state,
        )
