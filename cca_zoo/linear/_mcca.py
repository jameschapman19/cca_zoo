from typing import Iterable, Union

import numpy as np
from scipy.linalg import block_diag, eigh
from sklearn.decomposition import PCA

from cca_zoo._base import _BaseModel
from cca_zoo._utils._checks import _process_parameter
from cca_zoo._utils._cross_correlation import cross_cov


class MCCA(_BaseModel):
    r"""
    A class used to fit a Regularised CCA (canonical ridge) model. This model adds a regularization term to the CCA objective function to avoid overfitting and improve stability. It uses PCA to perform the optimization efficiently for high dimensional data.

    The objective function of regularised CCA is:

    .. math::

        w_{opt}=\underset{w}{\mathrm{argmax}}\{ w_1^TX_1^TX_2w_2  \}\\

        \text{subject to:}

        (1-c_1)w_1^TX_1^TX_1w_1+c_1w_1^Tw_1=n

        (1-c_2)w_2^TX_2^TX_2w_2+c_2w_2^Tw_2=n

    where :math:`c_i` are the regularization parameters for each view.

    Parameters
    ----------
    latent_dimensions : int, optional
        Number of latent dimensions to use, by default 1
    copy_data : bool, optional
        Whether to copy the data, by default True
    random_state : int, optional
        Random state, by default None
    c : Union[Iterable[float], float], optional
        Regularisation parameter, by default None
    accept_sparse : Union[bool, str], optional
        Whether to accept sparse data, by default None

    Examples
    --------
    >>> import numpy as np
    >>> rng=np.random.RandomState(0)
    >>> X1 = rng.random((10,5))
    >>> X2 = rng.random((10,5))
    >>> model = MCCA()
    >>> model.fit((X1,X2)).score((X1,X2))

    References
    --------
    Vinod, Hrishikesh _D. "Canonical ridge and econometrics of joint production." Journal of econometrics 4.2 (1976): 147-166.
    """

    def __init__(
        self,
        latent_dimensions: int = 1,
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
            latent_dimensions=latent_dimensions,
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
        views = self._validate_data(views)
        # Check the parameters
        self._check_params()
        views = self._process_data(views, **kwargs)
        eigvals, eigvecs = self._solve_gevp(views, y=y, **kwargs)
        # Compute the weights_ for each view
        self._weights(eigvals, eigvecs, views, **kwargs)
        # delete pca to save memory
        if self.pca:
            del self.pca_models
        return self

    def _process_data(self, views, **kwargs):
        if self.pca:
            views = self._apply_pca(views)
        return views

    def _solve_gevp(self, views: Iterable[np.ndarray], y=None, **kwargs):
        # Setup the eigenvalue problem
        C = self._C(views, **kwargs)
        D = self._D(views, **kwargs)
        self.splits = np.cumsum([view.shape[1] for view in views])
        # Solve the eigenvalue problem
        # Get the dimension of _C
        p = C.shape[0]
        # Solve the generalized eigenvalue problem Cx=lambda Dx using a subset of eigenvalues and eigenvectors
        [eigvals, eigvecs] = eigh(
            C,
            D,
            subset_by_index=[p - self.latent_dimensions, p - 1],
        )
        # Sort the eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigvals, axis=0)[::-1]
        if eigvals.shape[0] < self.latent_dimensions:
            [eigvals, eigvecs] = eigh(
                C,
                D,
            )
            # Sort the eigenvalues and eigenvectors in descending order
            idx = np.argsort(eigvals, axis=0)[::-1][: self.latent_dimensions]
        eigvecs = eigvecs[:, idx].real
        eigvals = eigvals[idx].real
        return eigvals, eigvecs

    def _weights(self, eigvals, eigvecs, views, **kwargs):
        # split eigvecs into weights_ for each view
        self.weights_ = np.split(eigvecs, self.splits[:-1], axis=0)
        if self.pca:
            # go from weights_ in PCA space to weights_ in original space
            self.weights_ = [
                pca.components_.T @ self.weights_[i]
                for i, pca in enumerate(self.pca_models)
            ]

    def _apply_pca(self, views):
        """
        Do data driven PCA on each view
        """
        self.pca_models = [PCA() for _ in views]
        # Fit PCA on each view
        return [self.pca_models[i].fit_transform(view) for i, view in enumerate(views)]

    def _C(self, views, **kwargs):
        all_views = np.hstack(views)
        C = np.cov(all_views, rowvar=False)
        C -= block_diag(*[np.cov(view, rowvar=False) for view in views])
        return C / len(views)

    def _D(self, views, **kwargs):
        if self.pca:
            # Can regularise by adding to diagonal
            D = block_diag(
                *[
                    np.diag((1 - self.c[i]) * pc.explained_variance_ + self.c[i])
                    for i, pc in enumerate(self.pca_models)
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

    Examples
    --------
    >>> import numpy as np
    >>> rng=np.random.RandomState(0)
    >>> X1 = rng.random((10,5))
    >>> X2 = rng.random((10,5))
    >>> model = rCCA(c=0.1)
    >>> model.fit((X1,X2)).score((X1,X2))


    References
    --------
    Vinod, Hrishikesh _D. "Canonical ridge and econometrics of joint production." Journal of econometrics 4.2 (1976): 147-166.
    """

    def _C(self, views, **kwargs):
        if len(views) != 2:
            raise ValueError(
                f"Model can only be used with two representations, but {len(views)} were given. Use MCCA or GCCA instead for CCA or MPLS for PLS."
            )
        if self.pca:
            # Compute the B matrices for each view
            B = [
                (1 - self.c[i]) * pc.explained_variance_ + self.c[i]
                for i, pc in enumerate(self.pca_models)
            ]
            C = cross_cov(
                views[0] / np.sqrt(B[0]), views[1] / np.sqrt(B[1]), rowvar=False
            )
            self.primary_view = 0
            return C @ C.T
        else:
            # cholesky decomposition of views
            self.L0 = np.linalg.inv(
                np.linalg.cholesky(
                    (1 - self.c[0]) * np.cov(views[0], rowvar=False)
                    + (self.c[0] + self.eps) * np.eye(views[0].shape[1])
                )
            )
            self.L1 = np.linalg.inv(
                np.linalg.cholesky(
                    (1 - self.c[1]) * np.cov(views[1], rowvar=False)
                    + (self.c[1] + self.eps) * np.eye(views[1].shape[1])
                )
            )
            C = cross_cov(views[0], views[1], rowvar=False)
            if views[0].shape[1] <= views[1].shape[1]:
                self.primary_view = 0
                self.T = self.L0 @ C @ self.L1 @ self.L1.T @ C.T @ self.L0.T
                return self.T
            else:
                self.primary_view = 1
                self.T = self.L1 @ C.T @ self.L0 @ self.L0.T @ C @ self.L1.T
                return self.T

    def _D(self, views, **kwargs):
        return None

    def _weights(self, eigvals, eigvecs, views):
        self.weights_ = [None] * 2
        if self.pca:
            B = [
                (1 - self.c[i]) * pc.singular_values_**2 / self.n_samples_ + self.c[i]
                for i, pc in enumerate(self.pca_models)
            ]
            C = np.cov(
                views[self.primary_view], views[1 - self.primary_view], rowvar=False
            )[
                0 : views[self.primary_view].shape[1],
                views[self.primary_view].shape[1] :,
            ]
            # Compute the weight matrix for primary view
            self.weights_[1 - self.primary_view] = (
                # Project view 1 onto its principal components
                self.pca_models[1 - self.primary_view].components_.T
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
            self.weights_[self.primary_view] = (
                # Project view 2 onto its principal components
                self.pca_models[self.primary_view].components_.T
                # Scale by the inverse of the square root of B[1]
                @ np.diag(1 / np.sqrt(B[self.primary_view]))
                # Multiply by the eigenvectors
                @ eigvecs
            )
        else:
            if self.primary_view == 0:
                self.weights_[0] = self.L0.T @ eigvecs
                self.weights_[1] = (
                    (self.L1.T @ self.L1)
                    @ cross_cov(views[1], views[0], rowvar=False)
                    @ self.weights_[0]
                )
            else:
                self.weights_[1] = self.L1.T @ eigvecs
                self.weights_[0] = (
                    (self.L0.T @ self.L0)
                    @ cross_cov(views[0], views[1], rowvar=False)
                    @ self.weights_[1]
                )

    def _more_tags(self):
        # Inherit all tags from MCCA but override the multiview tag
        tags = super()._more_tags()
        tags["multiview"] = False
        return tags


class CCA(rCCA):
    r"""
    A class used to fit a simple CCA model. This model finds the linear projections of two representations that maximize their correlation.

    The objective function of CCA is:

    .. math::

        w_{opt}=\underset{w}{\mathrm{argmax}}\{ w_1^TX_1^TX_2w_2  \}\\

        \text{subject to:}

        w_1^TX_1^TX_1w_1=n

        w_2^TX_2^TX_2w_2=n

    Parameters
    ----------
    latent_dimensions : int, optional
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
    >>> import numpy as np
    >>> rng=np.random.RandomState(0)
    >>> X1 = rng.random((10,5))
    >>> X2 = rng.random((10,5))
    >>> model = CCA()
    >>> model.fit((X1,X2)).score((X1,X2))
    """

    def __init__(
        self,
        latent_dimensions: int = 1,
        copy_data=True,
        random_state=None,
        accept_sparse=None,
        eps: float = 1e-6,
        pca: bool = True,
    ):
        # Initialize the rCCA class with c set to 0
        super().__init__(
            latent_dimensions=latent_dimensions,
            copy_data=copy_data,
            random_state=random_state,
            c=0,  # Setting c to 0
            accept_sparse=accept_sparse,
            eps=eps,
            pca=pca,
        )
