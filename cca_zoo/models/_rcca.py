from typing import Iterable, Union

import numpy as np
from scipy.linalg import eigh
from sklearn.decomposition import PCA

from cca_zoo.models._base import BaseModel
from cca_zoo.models._plsmixin import PLSMixin
from cca_zoo.utils.check_values import _process_parameter


class rCCA(BaseModel):
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

    Example
    -------
    >>> from cca_zoo.models import rCCA
    >>> import numpy as np
    >>> rng=np.random.RandomState(0)
    >>> X1 = rng.random((10,5))
    >>> X2 = rng.random((10,5))
    >>> model = rCCA(c=[0.1,0.1])
    >>> model.fit((X1,X2)).score((X1,X2))
    array([0.95222128])
    """

    def __init__(
        self,
        latent_dims: int = 1,
        copy_data=True,
        random_state=None,
        c: Union[Iterable[float], float] = None,
        accept_sparse=None,
        eps: float = 1e-6,
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

    def _check_params(self):
        # Process the c parameter for each view
        self.c = _process_parameter("c", self.c, 0, self.n_views_)

    def fit(self, views: Iterable[np.ndarray], y=None, **kwargs):
        # Validate the input data
        self._validate_data(views)
        # Check the parameters
        self._check_params()
        # Setup the eigenvalue problem
        C, D = self._setup_evp(views, **kwargs)
        # Solve the eigenvalue problem
        eigvals, eigvecs = self._solve_evp(C, D)
        # Compute the weights for each view
        self._weights(eigvals, eigvecs, views)
        return self

    def _setup_evp(self, views: Iterable[np.ndarray], **kwargs):
        # Get the number of samples
        n = views[0].shape[0]
        # Perform PCA on each view
        self.principal_components = _pca_data(*views)
        # Compute the B matrices for each view
        self.Bs = [
            (1 - self.c[i]) * pc.singular_values_**2 / n + self.c[i]
            for i, pc in enumerate(self.principal_components)
        ]
        # Compute the C and D matrices for two views
        C, D = self._two_view_evp(views)
        return C, D

    def _weights(self, eigvals, eigvecs, views):
        # Get the R and B matrices for each view
        R, B = self._get_R_B(views)
        # Compute the cross-covariance matrix between R[0] and R[1]
        R_12 = np.cov(R[0], R[1], rowvar=False)[0 : R[0].shape[1], R[0].shape[1] :]

        # Compute the weight matrix for view 1
        w_x = (
            # Project view 1 onto its principal components
            self.principal_components[0].components_.T
            # Scale by the inverse of B[0]
            @ np.diag(1 / B[0])
            # Multiply by the cross-covariance matrix
            @ R_12
            # Scale by the inverse of the square root of B[1]
            @ np.diag(1 / np.sqrt(B[1]))
            # Multiply by the eigenvectors
            @ eigvecs
            # Scale by the inverse of the square root of eigenvalues
            / np.sqrt(eigvals)
        )

        # Compute the weight matrix for view 2
        w_y = (
            # Project view 2 onto its principal components
            self.principal_components[1].components_.T
            # Scale by the inverse of the square root of B[1]
            @ np.diag(1 / np.sqrt(B[1]))
            # Multiply by the eigenvectors
            @ eigvecs
        )

        # Store the weight matrices as a list
        self.weights = [w_x, w_y]

    def _get_R_B(self, views):
        # Get the number of samples
        n = views[0].shape[0]
        # Compute the B matrices for each view
        B = [
            (1 - self.c[i]) * pc.singular_values_**2 / n + self.c[i]
            for i, pc in enumerate(self.principal_components)
        ]
        # Compute the R matrices for each view by projecting them onto their principal components
        R = [pc.transform(view) for view, pc in zip(views, self.principal_components)]
        return R, B

    def _solve_evp(self, C, D=None):
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

    def _two_view_evp(self, views):
        # Get the R and B matrices for each view
        R, B = self._get_R_B(views)
        # Compute the cross-covariance matrix between R[0] and R[1]
        R_12 = np.cov(R[0], R[1], rowvar=False)[0 : R[0].shape[1], R[0].shape[1] :]
        # Compute the M matrix as a function of R_12 and B
        M = (
            np.diag(1 / np.sqrt(B[1]))
            @ R_12.T
            @ np.diag(1 / B[0])
            @ R_12
            @ np.diag(1 / np.sqrt(B[1]))
        )
        return M, None

    def _more_tags(self):
        # Indicate that this class is for multiview data
        return {"multiview": True}


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


class PLS(rCCA, PLSMixin):
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

    Example
    -------

    >>> from cca_zoo.models import PLS
    >>> import numpy as np
    >>> rng=np.random.RandomState(0)
    >>> X1 = rng.random((10,5))
    >>> X2 = rng.random((10,5))
    >>> model = PLS()
    >>> model.fit((X1,X2)).score((X1,X2))
    array([0.81796873])
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


def _pca_data(*views: np.ndarray):
    """
    Performs PCA on the data and returns the scores and loadings

    Parameters
    ----------
    views : np.ndarray

    Returns
    -------
    Us : list of np.ndarray
        The loadings for each view
    Ss : list of np.ndarray
        The scores for each view
    Vs : list of np.ndarray
        The eigenvectors for each view

    """
    principal_components = []
    for view in views:
        principal_components.append(PCA().fit(view))
    return principal_components
