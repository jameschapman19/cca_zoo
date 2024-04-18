import itertools
from abc import abstractmethod
from typing import Iterable, Union, List, Optional, Any

import numpy as np
from numpy.linalg import svd
from scipy.linalg import block_diag
from sklearn.base import BaseEstimator, MultiOutputMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array

from cca_zoo._utils._cross_correlation import cross_corrcoef


class _BaseModel(BaseEstimator, MultiOutputMixin, TransformerMixin):
    """
    A base class for multivariate latent variable linear.

    This class implements common methods and attributes for fitting and transforming
    multiple representations of data using latent variable linear. It inherits from scikit-learn's
    BaseEstimator, MultiOutputMixin and RegressorMixin classes.

    Parameters
    ----------
    latent_dimensions: int, optional
        Number of latent dimensions to fit. Default is 1.
    copy_data: bool, optional
        Whether to copy the data. Default is True.
    accept_sparse: bool, optional
        Whether to accept sparse data. Default is False.
    random_state: int, RandomState instance or None, optional (default=None)
        Pass an int for reproducible output across multiple function calls.

    Attributes
    ----------
    n_views_: int
        Number of representations.
    n_features_in_: list of int
        Number of features for each view.
    weights_: list of numpy arrays
        Weight vectors for each view.
    """

    weights_ = None

    def __init__(
        self,
        latent_dimensions: int = 1,
        copy_data=True,
        accept_sparse=False,
        random_state: Union[int, np.random.RandomState] = None,
    ):
        self.latent_dimensions = latent_dimensions
        self.copy_data = copy_data
        self.accept_sparse = accept_sparse
        self.random_state = random_state

    def _validate_data(self, views: Iterable[np.ndarray]):
        if not self._get_tags().get("multiview", False) and len(views) > 2:
            raise ValueError(
                f"Model can only be used with two representations, but {len(views)} were given. "
                "Use MCCA or GCCA instead for CCA or MPLS for PLS."
            )
        if self.copy_data:
            views = [
                check_array(
                    view,
                    copy=True,
                    accept_sparse=False,
                    accept_large_sparse=False,
                    ensure_min_samples=max(2, self.latent_dimensions),
                    ensure_min_features=self.latent_dimensions,
                )
                for view in views
            ]
        else:
            views = [
                check_array(
                    view,
                    copy=False,
                    accept_sparse=False,
                    accept_large_sparse=False,
                    ensure_min_samples=max(2, self.latent_dimensions),
                    ensure_min_features=self.latent_dimensions,
                )
                for view in views
            ]
        if not all(view.shape[0] == views[0].shape[0] for view in views):
            raise ValueError("All representations must have the same number of samples")
        if not all(view.ndim == 2 for view in views):
            raise ValueError("All representations must have 2 dimensions")
        self.n_views_ = len(views)
        self.n_features_in_ = [view.shape[1] for view in views]
        self.n_samples_ = views[0].shape[0]
        return views

    def _check_params(self):
        """
        Checks the parameters of the model.
        """
        pass

    @abstractmethod
    def fit(self, views: Iterable[np.ndarray], y=None, **kwargs):
        """
        Fits the model to the given data

        Parameters
        ----------
        views: list/tuple of numpy arrays or array likes with the same number of rows (samples)
        y: None
        kwargs: any additional keyword arguments required by the given model

        Returns
        -------
        self: object

        """
        return self

    def transform(
        self, views: Iterable[np.ndarray], *args, **kwargs
    ) -> List[np.ndarray]:
        """
        Transforms the given representations using the fitted model.

        Parameters
        ----------
        views: list/tuple of numpy arrays or array likes with the same number of rows (samples)
        kwargs: any additional keyword arguments required by the given model

        Returns
        -------
        representations: list of numpy arrays

        """
        check_is_fitted(self)
        views = [
            check_array(
                view,
                copy=True,
                accept_sparse=False,
                accept_large_sparse=False,
            )
            for view in views
        ]
        representations = []
        for i, view in enumerate(views):
            representation = view @ self.weights_[i]
            representations.append(representation)
        return representations

    def pairwise_correlations(
        self, views: Iterable[np.ndarray], **kwargs
    ) -> np.ndarray:
        """
        Calculate pairwise correlations between representations in each dimension.

        Parameters
        ----------
        views: list/tuple of numpy arrays or array-like objects with the same number of rows (samples)
        kwargs: any additional keyword arguments required by the given model

        Returns
        -------
        pairwise_correlations: numpy array of shape (n_views, n_views, latent_dimensions)
        """
        representations = self.transform(views, **kwargs)
        all_corrs = []
        for x, y in itertools.product(representations, repeat=2):
            all_corrs.append(np.diag(cross_corrcoef(x.T, y.T)))
        all_corrs = np.array(all_corrs).reshape(
            (self.n_views_, self.n_views_, self.latent_dimensions)
        )
        return all_corrs

    def average_pairwise_correlations(
        self, views: Iterable[np.ndarray], **kwargs
    ) -> np.ndarray:
        """
        Calculate the average pairwise correlations between representations in each dimension.

        Parameters
        ----------
        views: list/tuple of numpy arrays or array-like objects with the same number of rows (samples)
        kwargs: any additional keyword arguments required by the given model

        Returns
        -------
        average_pairwise_correlations: numpy array of shape (latent_dimensions, )
        """
        pair_corrs = self.pairwise_correlations(views, **kwargs)
        # Sum all the pairwise correlations for each dimension, subtract self-correlations, and divide by the number of representations
        dim_corrs = np.sum(pair_corrs, axis=(0, 1)) - pair_corrs.shape[0]
        # Number of pairs is n_views choose 2
        num_pairs = (self.n_views_ * (self.n_views_ - 1)) / 2
        dim_corrs = dim_corrs / (2 * num_pairs)
        return dim_corrs

    def score(
        self, views: Iterable[np.ndarray], y: Optional[Any] = None, **kwargs
    ) -> float:
        """
        Calculate the sum of average pairwise correlations between representations.

        Parameters
        ----------
        views: list/tuple of numpy arrays or array-like objects with the same number of rows (samples)
        y: None
        kwargs: any additional keyword arguments required by the given model

        Returns
        -------
        score: float
            Sum of average pairwise correlations between representations.
        """
        return self.average_pairwise_correlations(views, **kwargs).sum()

    def loadings_(self, views: Iterable[np.ndarray], **kwargs) -> List[np.ndarray]:
        """
        Calculate canonical loadings for each view.

        Canonical loadings represent the correlation between the original variables
        in a view and their respective canonical variates. Canonical variates are
        linear combinations of the original variables formed to maximize the correlation
        with canonical variates from another view.

        Mathematically, given two representations \(X_i\), canonical variates
        from the representations are:

            \(Z_i = w_i^T X_i\)

        The canonical loading for a variable in \(X_i\) is the correlation between
        that variable and \(Z_i\).

        Parameters
        ----------
        views: list/tuple of numpy arrays
            Each array corresponds to a view. All representations must have the same number of rows (observations).

        Returns
        -------
        loadings_: list of numpy arrays
            Canonical loadings for each view. High absolute values indicate that
            the respective original variables play a significant role in defining the canonical variate.

        """
        check_is_fitted(self, attributes=["weights_"])
        representations = self.transform(views, **kwargs)
        loadings = [
            cross_corrcoef(view, representation, rowvar=False)
            for view, representation in zip(views, representations)
        ]
        return loadings

    def explained_variance(self, views: Iterable[np.ndarray]) -> List[np.ndarray]:
        """
        Calculates variance captured by each latent dimension for each view.

        Returns
        -------
        variances_by_dimension: list of numpy arrays
        """
        check_is_fitted(self, attributes=["weights_"])

        normalized_weights_ = [
            weight / np.linalg.norm(weight, axis=0) for weight in self.weights_
        ]

        # Transform views using normalized weights
        transformed_views = [
            view @ weights for view, weights in zip(views, normalized_weights_)
        ]

        # Calculate variance for each latent dimension
        variances_by_dimension = [
            np.var(transformed_view, axis=0) for transformed_view in transformed_views
        ]
        return variances_by_dimension

    def explained_variance_ratio(self, views: Iterable[np.ndarray]) -> List[np.ndarray]:
        """
        Calculates variance ratio captured by each latent dimension for each view.

        Returns
        -------
        variance_ratios: list of numpy arrays
        """
        total_variances = [
            np.sum(s**2) / (view.shape[0] - 1)
            for view in views
            for _, s, _ in [svd(view)]
        ]

        variances_by_dimension = self.explained_variance(views)

        # Calculate variance ratio for each dimension
        variance_ratios = [
            var_by_dim / total_var
            for var_by_dim, total_var in zip(variances_by_dimension, total_variances)
        ]
        return variance_ratios

    def explained_variance_cumulative(
        self, views: Iterable[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Calculates cumulative explained variance ratio for each latent dimension.

        Returns
        -------
        cumulative_variance_ratios: list of numpy arrays
        """
        variance_ratios = self.explained_variance_ratio(views)
        cumulative_variance_ratios = [np.cumsum(ratio) for ratio in variance_ratios]
        return cumulative_variance_ratios

    def _compute_covariance(self, views: Iterable[np.ndarray]) -> np.ndarray:
        """
        Computes the covariance matrix for the given representations.

        Parameters
        ----------
        views: list/tuple of numpy arrays or array likes with the same number of rows (samples)

        Returns
        -------
        cov: numpy array
            Computed covariance matrix.
        """
        cov = np.cov(np.hstack(views), rowvar=False)
        cov -= block_diag(*[np.cov(view, rowvar=False) for view in views])
        return cov

    def explained_covariance(self, views: Iterable[np.ndarray]) -> np.ndarray:
        """
        Calculates the covariance matrix of the transformed components for each view.

        Parameters
        ----------
        views: list/tuple of numpy arrays or array likes with the same number of rows (samples)

        Returns
        -------
        explained_covariances: list of numpy arrays
            Covariance matrices for the transformed components of each view.
        """
        check_is_fitted(self, attributes=["weights_"])

        # Transform the representations using the loadings_
        representations = [
            view @ loading for view, loading in zip(views, self.loadings_(views))
        ]

        k = representations[0].shape[1]

        explained_covariances = np.zeros(k)

        # just take the kth column of each transformed view and _compute_covariance
        for i in range(k):
            representations_k = [view[:, i][:, None] for view in representations]
            cov_ = self._compute_covariance(representations_k)
            _, s_, _ = svd(cov_)
            explained_covariances[i] = s_[0]

        return explained_covariances

    def explained_covariance_ratio(self, views: Iterable[np.ndarray]) -> np.ndarray:
        # only works for 2 views
        check_is_fitted(self, attributes=["weights_"])
        assert len(views) == 2, "Only works for 2 views"
        minimum_dimension = min([view.shape[1] for view in views])
        cov = self._compute_covariance(views)
        _, S, _ = svd(cov)
        # select every other element starting from the first until the minimum dimension
        total_explained_covariance = S[::2][:minimum_dimension].sum()
        explained_covariances = self.explained_covariance(views)
        explained_covariance_ratios = explained_covariances / total_explained_covariance
        return explained_covariance_ratios

    def explained_covariance_cumulative(
        self, views: Iterable[np.ndarray]
    ) -> np.ndarray:
        """
        Calculates the cumulative explained covariance ratio for each latent dimension for each view.

        Returns
        -------
        cumulative_ratios: list of numpy arrays
        """
        ratios = self.explained_covariance_ratio(views)
        cumulative_ratios = [np.cumsum(ratio) for ratio in ratios]

        return cumulative_ratios
