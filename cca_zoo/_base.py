import itertools
from abc import abstractmethod
from typing import Iterable, Union, List, Optional, Any

import numpy as np
from numpy.linalg import svd
from scipy.linalg import block_diag
from sklearn.base import BaseEstimator, MultiOutputMixin, RegressorMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted, check_array


class BaseModel(BaseEstimator, MultiOutputMixin, RegressorMixin):
    """
    A base class for multivariate latent variable linear.

    This class implements common methods and attributes for fitting and transforming
    multiple views of data using latent variable linear. It inherits from scikit-learn's
    BaseEstimator, MultiOutputMixin and RegressorMixin classes.

    Parameters
    ----------
    latent_dimensions : int, optional
        Number of latent dimensions to fit. Default is 1.
    copy_data : bool, optional
        Whether to copy the data. Default is True.
    accept_sparse : bool, optional
        Whether to accept sparse data. Default is False.
    random_state : int, RandomState instance or None, optional (default=None)
        Pass an int for reproducible output across multiple function calls.

    Attributes
    ----------
    n_views_ : int
        Number of views.
    n_features_ : list of int
        Number of features for each view.

    """

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
        self.random_state = check_random_state(random_state)
        self.dtypes = FLOAT_DTYPES
        self._loadings = None

    def _validate_data(self, views: Iterable[np.ndarray]):
        if not all(view.shape[0] == views[0].shape[0] for view in views):
            raise ValueError("All views must have the same number of samples")
        if not all(view.ndim == 2 for view in views):
            raise ValueError("All views must have 2 dimensions")
        if not all(view.dtype in self.dtypes for view in views):
            raise ValueError("All views must have dtype of {}.".format(self.dtypes))
        if not all(view.shape[1] >= self.latent_dimensions for view in views):
            raise ValueError(
                "All views must have at least {} features.".format(
                    self.latent_dimensions
                )
            )
        self.n_views_ = len(views)
        self.n_features_ = [view.shape[1] for view in views]
        self.n_samples_ = views[0].shape[0]
        if self.copy_data:
            return [check_array(view, copy=True, dtype=self.dtypes) for view in views]
        else:
            return [check_array(view, copy=False, dtype=self.dtypes) for view in views]

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
        views : list/tuple of numpy arrays or array likes with the same number of rows (samples)
        y : None
        kwargs: any additional keyword arguments required by the given model

        Returns
        -------
        self : object

        """
        return self

    def transform(self, views: Iterable[np.ndarray], **kwargs) -> List[np.ndarray]:
        """

        Parameters
        ----------
        views : list/tuple of numpy arrays or array likes with the same number of rows (samples)
        kwargs : any additional keyword arguments required by the given model

        Returns
        -------
        transformed_views : list of numpy arrays

        """
        check_is_fitted(self, attributes=["weights"])
        transformed_views = []
        for i, view in enumerate(views):
            transformed_view = view @ self.weights[i]
            transformed_views.append(transformed_view)
        return transformed_views

    def fit_transform(self, views: Iterable[np.ndarray], **kwargs) -> List[np.ndarray]:
        """
        Fits the model to the given data and returns the transformed views

        Parameters
        ----------
        views : list/tuple of numpy arrays or array likes with the same number of rows (samples)
        kwargs : any additional keyword arguments required by the given model

        Returns
        -------
        transformed_views : list of numpy arrays

        """
        return self.fit(views, **kwargs).transform(views, **kwargs)

    def pairwise_correlations(
        self, views: Iterable[np.ndarray], **kwargs
    ) -> np.ndarray:
        """
        Returns the pairwise correlations between the views in each dimension

        Parameters
        ----------
        views : list/tuple of numpy arrays or array likes with the same number of rows (samples)
        kwargs : any additional keyword arguments required by the given model

        Returns
        -------
        pairwise_correlations : numpy array of shape (n_views, n_views, latent_dimensions)

        """
        transformed_views = self.transform(views, **kwargs)
        all_corrs = []
        for x, y in itertools.product(transformed_views, repeat=2):
            all_corrs.append(
                np.diag(
                    np.corrcoef(x.T, y.T)[
                        : self.latent_dimensions, self.latent_dimensions :
                    ]
                )
            )
        try:
            all_corrs = np.array(all_corrs).reshape(
                (self.n_views_, self.n_views_, self.latent_dimensions)
            )
        except:
            print()
        return all_corrs

    def score(
        self, views: Iterable[np.ndarray], y: Optional[Any] = None, **kwargs
    ) -> float:
        """
        Returns the average pairwise correlation between the views

        Parameters
        ----------
        views : list/tuple of numpy arrays or array likes with the same number of rows (samples)
        y : None
        kwargs : any additional keyword arguments required by the given model

        Returns
        -------
        score : float


        """
        # by default return the average pairwise correlation in each dimension (for 2 views just the correlation)
        pair_corrs = self.pairwise_correlations(views, **kwargs)
        # sum all the pairwise correlations for each dimension. Subtract the self correlations. Divide by the number of views. Gives average correlation
        dim_corrs = np.sum(pair_corrs, axis=(0, 1)) - pair_corrs.shape[0]
        # number of pairs is n_views choose 2
        num_pairs = (self.n_views_ * (self.n_views_ - 1)) / 2
        dim_corrs = dim_corrs / (2 * num_pairs)
        return dim_corrs

    def canonical_loadings(
        self, views: Iterable[np.ndarray], normalize: bool = True, **kwargs
    ) -> List[np.ndarray]:
        """
        Calculate canonical loadings for each view.

        Canonical loadings represent the correlation between the original variables
        in a view and their respective canonical variates. Canonical variates are
        linear combinations of the original variables formed to maximize the correlation
        with canonical variates from another view.

        Mathematically, given two views \(X_i\), canonical variates
        from the views are:

            \(Z_i = w_i^T X_i\)

        The canonical loading for a variable in \(X_i\) is the correlation between
        that variable and \(Z_i\).

        Parameters
        ----------
        views : list/tuple of numpy arrays
            Each array corresponds to a view. All views must have the same number of rows (observations).

        Returns
        -------
        loadings : list of numpy arrays
            Canonical loadings for each view. High absolute values indicate that
            the respective original variables play a significant role in defining the canonical variate.

        """
        transformed_views = self.transform(views, **kwargs)
        if normalize:
            loadings = [
                np.corrcoef(view, transformed_view, rowvar=False)[
                    : view.shape[1], view.shape[1] :
                ]
                for view, transformed_view in zip(views, transformed_views)
            ]
        else:
            loadings = [
                view.T @ transformed_view
                for view, transformed_view in zip(views, transformed_views)
            ]
        return loadings

    @property
    def loadings(self) -> List[np.ndarray]:
        """
        Compute and return loadings for each view. These are cached for performance optimization.

        In the context of the cca-zoo models, loadings are the normalized weights. Due to the structure of these models,
        weight vectors are normalized such that w'X'Xw = 1, as opposed to w'w = 1, which is commonly used in PCA.
        As a result, when computing the loadings, the weights are normalized to have unit norm, ensuring that the loadings
        range between -1 and 1.

        It's essential to differentiate between these loadings and canonical loadings. The latter are correlations between
        the original variables and their corresponding canonical variates.

        Returns
        -------
        List[np.ndarray]
            Loadings for each view.
        """
        check_is_fitted(self, attributes=["weights"])
        if self._loadings is None:
            # Compute loadings only if they haven't been computed yet
            self._loadings = [
                weights / np.linalg.norm(weights, axis=0) for weights in self.weights
            ]
        return self._loadings

    def explained_variance(self, views: Iterable[np.ndarray]) -> List[np.ndarray]:
        """
        Calculates the variance captured by each latent dimension for each view.

        Returns
        -------
        transformed_vars : list of numpy arrays
        """
        check_is_fitted(self, attributes=["weights"])

        # Transform the views using the loadings
        transformed_views = [
            view @ loading for view, loading in zip(views, self.loadings)
        ]

        # Calculate the variance of each latent dimension in the transformed views
        transformed_vars = [
            np.var(transformed, axis=0) for transformed in transformed_views
        ]

        return transformed_vars

    def explained_variance_ratio(self, views: Iterable[np.ndarray]) -> List[np.ndarray]:
        """
        Calculates the ratio of the variance captured by each latent dimension to the total variance for each view.

        Returns
        -------
        explained_variance_ratios : list of numpy arrays
        """
        total_vars = [
            (np.sum(s**2) / (view.shape[0] - 1))
            for view in views
            for _, s, _ in [svd(view)]
        ]

        transformed_vars = self.explained_variance(views)

        # Calculate the explained variance ratio for each latent dimension for each view
        explained_variance_ratios = [
            transformed_var / total_var
            for transformed_var, total_var in zip(transformed_vars, total_vars)
        ]

        return explained_variance_ratios

    def explained_variance_cumulative(
        self, views: Iterable[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Calculates the cumulative explained variance ratio for each latent dimension for each view.

        Returns
        -------
        cumulative_ratios : list of numpy arrays
        """
        ratios = self.explained_variance_ratio(views)
        cumulative_ratios = [np.cumsum(ratio) for ratio in ratios]

        return cumulative_ratios

    def _compute_covariance(self, views: Iterable[np.ndarray]) -> np.ndarray:
        """
        Computes the covariance matrix for the given views.

        Parameters
        ----------
        views : list/tuple of numpy arrays or array likes with the same number of rows (samples)

        Returns
        -------
        cov : numpy array
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
        views : list/tuple of numpy arrays or array likes with the same number of rows (samples)

        Returns
        -------
        explained_covariances : list of numpy arrays
            Covariance matrices for the transformed components of each view.
        """
        check_is_fitted(self, attributes=["weights"])

        # Transform the views using the loadings
        transformed_views = [
            view @ loading for view, loading in zip(views, self.loadings)
        ]

        k = transformed_views[0].shape[1]

        explained_covariances = np.zeros(k)

        # just take the kth column of each transformed view and _compute_covariance
        for i in range(k):
            transformed_views_k = [view[:, i][:, None] for view in transformed_views]
            cov_ = self._compute_covariance(transformed_views_k)
            _, s_, _ = svd(cov_)
            explained_covariances[i] = s_[0]

        return explained_covariances

    def explained_covariance_ratio(self, views: Iterable[np.ndarray]) -> np.ndarray:
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
        cumulative_ratios : list of numpy arrays
        """
        ratios = self.explained_covariance_ratio(views)
        cumulative_ratios = [np.cumsum(ratio) for ratio in ratios]

        return cumulative_ratios

    def predict(self, views: Iterable[np.ndarray]) -> List[np.ndarray]:
        """
        Predicts the missing view from the given views.


        Parameters
        ----------
        views : list/tuple of numpy arrays or array likes with the same number of rows (samples)

        Returns
        -------
        predicted_views : list of numpy arrays. None if the view is missing.
            Predicted views.

        Examples
        --------
        >>> import numpy as np
        >>> X1 = np.random.rand(100, 5)
        >>> X2 = np.random.rand(100, 5)
        >>> cca = CCA()
        >>> cca.fit([X1, X2])
        >>> X1_pred, X2_pred = cca.predict([X1, None])

        """
        check_is_fitted(self, attributes=["weights"])
        # check if views is same length as weights
        if len(views) != len(self.weights):
            raise ValueError(
                "The number of views must be the same as the number of weights. Put None for missing views."
            )
        transformed_views = []
        for i, view in enumerate(views):
            if view is not None:
                transformed_view = view @ self.weights[i]
                transformed_views.append(transformed_view)
        # average the transformed views
        average_score = np.mean(transformed_views, axis=0)
        # return the average score transformed back to the original space
        reconstucted_views = []
        for i, view in enumerate(views):
            reconstructed_view = average_score @ np.linalg.pinv(self.weights[i])
            reconstucted_views.append(reconstructed_view)
        return reconstucted_views
