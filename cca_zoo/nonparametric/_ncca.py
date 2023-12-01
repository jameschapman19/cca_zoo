from typing import Iterable, Union

import numpy as np
from sklearn.metrics import pairwise_kernels
from sklearn.neighbors import NearestNeighbors

from cca_zoo._base import _BaseModel
from cca_zoo._utils._checks import _process_parameter


class NCCA(_BaseModel):
    """
    A class used to fit nonparametric (NCCA) model. This model extends CCA to nonlinear relationships by using local linear projections based on nearest neighbors.

    Parameters
    ----------
    latent_dimensions : int, optional
        Number of latent dimensions to use, by default 1
    copy_data : bool, optional
        Whether to copy the data, by default True
    accept_sparse : bool, optional
        Whether to accept sparse data as input, by default False
    random_state : Union[int, np.random.RandomState], optional
        Random seed for reproducibility, by default None
    nearest_neighbors : int, optional
        Number of nearest neighbors to use for local linear projections, by default None. If None, it will use the square root of the number of samples.
    gamma : Iterable[float], optional
        Bandwidth parameter or list of parameters for the RBF kernel for each view, by default None. If None, it will use the median heuristic.


    References
    ----------
    Michaeli, Tomer, Weiran Wang, and Karen Livescu. "Nonparametric canonical correlation analysis." International conference on machine learning. PMLR, 2016.

    Example
    -------
    >>> from cca_zoo.linear import NCCA
    >>> X1 = np.random.rand(10,5)
    >>> X2 = np.random.rand(10,5)
    >>> model = NCCA()
    >>> model.fit((X1,X2)).score((X1,X2))
    array([1.])
    """

    def __init__(
        self,
        latent_dimensions: int = 1,
        copy_data=True,
        accept_sparse=False,
        random_state: Union[int, np.random.RandomState] = None,
        nearest_neighbors=None,
        gamma: Iterable[float] = None,
    ):
        # Call the parent class constructor
        super().__init__(
            latent_dimensions=latent_dimensions,
            copy_data=copy_data,
            accept_sparse=accept_sparse,
            random_state=random_state,
        )
        # Store the nearest neighbors and gamma parameters
        self.nearest_neighbors = nearest_neighbors
        self.gamma = gamma

    def _check_params(self):
        # Process the nearest neighbors and gamma parameters for each view
        self.nearest_neighbors = _process_parameter(
            "nearest_neighbors", self.nearest_neighbors, 1, self.n_views_
        )
        self.gamma = _process_parameter("gamma", self.gamma, None, self.n_views_)
        # Use RBF kernel as default for each view
        self.kernel = _process_parameter("kernel", None, "rbf", self.n_views_)

    def fit(self, views: Iterable[np.ndarray], y=None, **kwargs):
        # Validate the input data
        views = self._validate_data(views)
        # Check the parameters
        self._check_params()
        # Store the training representations
        self.train_views = views
        # Fit a nearest neighbors model for each view
        self.knns = [
            NearestNeighbors(n_neighbors=self.nearest_neighbors[i]).fit(view)
            for i, view in enumerate(views)
        ]
        # Find the nearest neighbors for each view
        NNs = [
            self.knns[i].kneighbors(view, self.nearest_neighbors[i])
            for i, view in enumerate(views)
        ]
        # Compute the kernel matrices for each view
        kernels = [self._get_kernel(i, view) for i, view in enumerate(self.train_views)]
        # Fill the weight matrices with the kernel values
        self.Ws = [fill_w(kernel, inds) for kernel, (dists, inds) in zip(kernels, NNs)]
        # Normalize the weight matrices by row and column sums
        self.Ws = [
            self.Ws[0] / self.Ws[0].sum(axis=1, keepdims=True),
            self.Ws[1] / self.Ws[1].sum(axis=0, keepdims=True),
        ]
        # Compute the cross-covariance matrix between the weight matrices
        S = self.Ws[0] @ self.Ws[1]
        # Perform singular value decomposition on the cross-covariance matrix
        U, S, Vt = np.linalg.svd(S)
        # Compute the canonical score vectors for each view
        self.f = U[:, 1 : self.latent_dimensions + 1] * np.sqrt(self.n_samples_)
        self.g = Vt[1 : self.latent_dimensions + 1, :].T * np.sqrt(self.n_samples_)
        # Store the canonical correlations
        self.S = S[1 : self.latent_dimensions + 1]
        return self

    def transform(self, views: Iterable[np.ndarray], **kwargs):
        # Find the nearest neighbors for each view
        nns = [
            self.knns[i].kneighbors(view, self.nearest_neighbors[i])
            for i, view in enumerate(views)
        ]
        # Compute the kernel matrices between the training and test representations
        kernels = [
            self._get_kernel(i, self.train_views[i], Y=view)
            for i, view in enumerate(views)
        ]
        # Fill the weight matrices with the kernel values
        Wst = [fill_w(kernel, inds) for kernel, (dists, inds) in zip(kernels, nns)]
        # Normalize the weight matrices by row sums
        Wst = [
            Wst[0] / Wst[0].sum(axis=1, keepdims=True),
            Wst[1] / Wst[1].sum(axis=1, keepdims=True),
        ]
        # Compute the cross-covariance matrix between the weight matrices and the training weight matrices
        St = [Wst[0] @ self.Ws[1], Wst[1] @ self.Ws[0]]
        # Project the cross-covariance matrix onto the canonical score vectors and normalize by the canonical correlations
        return St[0] @ self.g * (1 / self.S), St[1] @ self.f * (1 / self.S)

    def _get_kernel(self, view, X, Y=None):
        # Get the gamma parameter for the RBF kernel
        params = {
            "gamma": self.gamma[view],
        }
        # Compute the pairwise kernel values between representations and Y using the specified kernel function and parameters
        return pairwise_kernels(
            X, Y, metric=self.kernel[view], filter_params=True, **params
        )


def fill_w(kernels, inds):
    # Create an empty matrix with the same shape as kernels
    w = np.zeros_like(kernels)
    # For each row, fill the corresponding columns with the kernel values
    for i, ind in enumerate(inds):
        w[ind, i] = kernels[ind, i]
        # Transpose the matrix to match the original orientation
    return w.T
