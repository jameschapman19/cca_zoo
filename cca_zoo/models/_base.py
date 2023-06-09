import itertools
from abc import abstractmethod
from typing import Iterable, Union

import numpy as np
from sklearn.base import BaseEstimator, MultiOutputMixin, RegressorMixin
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted, check_random_state


class BaseModel(BaseEstimator, MultiOutputMixin, RegressorMixin):
    """
    A base class for multivariate latent variable models.

    This class implements common methods and attributes for fitting and transforming
    multiple views of data using latent variable models. It inherits from scikit-learn's
    BaseEstimator, MultiOutputMixin and RegressorMixin classes.

    Parameters
    ----------
    latent_dims : int, optional
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
        latent_dims: int = 1,
        copy_data=True,
        accept_sparse=False,
        random_state: Union[int, np.random.RandomState] = None,
    ):
        self.latent_dims = latent_dims
        self.copy_data = copy_data
        self.accept_sparse = accept_sparse
        self.random_state = check_random_state(random_state)
        self.dtypes = FLOAT_DTYPES

    def _validate_data(
        self,
        views: Iterable[np.ndarray],
    ):
        if not all(view.shape[0] == views[0].shape[0] for view in views):
            raise ValueError("All views must have the same number of samples")
        if not all(view.ndim == 2 for view in views):
            raise ValueError("All views must have 2 dimensions")
        if not all(view.dtype in self.dtypes for view in views):
            raise ValueError("All views must have dtype of {}.".format(self.dtypes))
        if not all(view.shape[1] >= self.latent_dims for view in views):
            raise ValueError(
                "All views must have at least {} features.".format(self.latent_dims)
            )
        self.n_views_ = len(views)
        self.n_features_ = [view.shape[1] for view in views]
        self.n_samples_ = views[0].shape[0]

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

    def transform(self, views: Iterable[np.ndarray], **kwargs):
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

    def fit_transform(self, views: Iterable[np.ndarray], **kwargs):
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

    def pairwise_correlations(self, views: Iterable[np.ndarray], **kwargs):
        """
        Returns the pairwise correlations between the views in each dimension

        Parameters
        ----------
        views : list/tuple of numpy arrays or array likes with the same number of rows (samples)
        kwargs : any additional keyword arguments required by the given model

        Returns
        -------
        pairwise_correlations : numpy array of shape (n_views, n_views, latent_dims)

        """
        transformed_views = self.transform(views, **kwargs)
        all_corrs = []
        for x, y in itertools.product(transformed_views, repeat=2):
            all_corrs.append(
                np.diag(np.corrcoef(x.T, y.T)[: self.latent_dims, self.latent_dims :])
            )
        all_corrs = np.array(all_corrs).reshape(
            (self.n_views_, self.n_views_, self.latent_dims)
        )
        return all_corrs

    def score(self, views: Iterable[np.ndarray], y=None, **kwargs):
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
        dim_corrs = (
            pair_corrs.sum(axis=tuple(range(pair_corrs.ndim - 1))) - self.n_views_
        ) / (self.n_views_**2 - self.n_views_)
        return dim_corrs

    def factor_loadings(self, views: Iterable[np.ndarray], normalize=True, **kwargs):
        """
        Returns the factor loadings for each view

        Parameters
        ----------
        views : list/tuple of numpy arrays or array likes with the same number of rows (samples)
        normalize : bool, optional
            Whether to normalize the factor loadings. Default is True.
        kwargs : any additional keyword arguments required by the given model

        Returns
        -------
        factor_loadings : list of numpy arrays

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
