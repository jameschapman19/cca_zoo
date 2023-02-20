import itertools
from abc import abstractmethod
from typing import Union, Iterable

import numpy as np
from sklearn.base import BaseEstimator, MultiOutputMixin, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_random_state, check_is_fitted, FLOAT_DTYPES

from cca_zoo.utils.check_values import _check_views


class _BaseCCA(BaseEstimator, MultiOutputMixin, RegressorMixin):
    """
    Base class for CCA methods.
    """

    def __init__(
        self,
        latent_dims: int = 1,
        scale=True,
        centre=True,
        copy_data=True,
        accept_sparse=False,
        random_state: Union[int, np.random.RandomState] = None,
    ):
        """
        Parameters
        ----------
        latent_dims : int, optional
            Number of latent dimensions to fit. Default is 1.
        scale : bool, optional
            Whether to scale the data to unit variance. Default is True.
        centre : bool, optional
            Whether to centre the data. Default is True.
        copy_data : bool, optional
            Whether to copy the data. Default is True.
        accept_sparse : bool, optional
            Whether to accept sparse data. Default is False.
        random_state : int, RandomState instance or None, optional (default=None)
            Pass an int for reproducible output across multiple function calls.

        """
        self.latent_dims = latent_dims
        self.scale = scale
        self.centre = centre
        self.copy_data = copy_data
        self.accept_sparse = accept_sparse
        self.random_state = check_random_state(random_state)
        self.dtypes = FLOAT_DTYPES

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
        _check_views(views)
        views = [self.scalers[i].transform(view) for i, view in enumerate(views)]
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

    def get_factor_loadings(
        self, views: Iterable[np.ndarray], normalize=True, **kwargs
    ):
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
            (len(transformed_views), len(transformed_views), self.latent_dims)
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
        # n views
        n_views = pair_corrs.shape[0]
        # sum all the pairwise correlations for each dimension. Subtract the self correlations. Divide by the number of views. Gives average correlation
        dim_corrs = (
            pair_corrs.sum(axis=tuple(range(pair_corrs.ndim - 1))) - n_views
        ) / (n_views**2 - n_views)
        return dim_corrs

    def _check_params(self):
        pass

    def _validate_inputs(self, views):
        """
        Checks that the input data is valid

        Parameters
        ----------
        views : list/tuple of numpy arrays or array likes with the same number of rows (samples)

        Returns
        -------
        views : list of numpy arrays

        """
        # if doesn't have tag multiview=True then check number of views equals 2
        if not self._get_tags().get("multiview", False):
            if len(views) != 2:
                raise ValueError("Only two views are supported for this model.")
        _check_views(views)
        views=[self._validate_data(view,accept_sparse=["csr", "csc", "coo"],dtype=self.dtypes, copy=self.copy_data) for view in views]

        self.scalers = [
            StandardScaler(
                copy=self.copy_data, with_mean=self.centre, with_std=self.scale
            )
            for _ in range(len(views))
        ]
        views = [
            scaler.fit_transform(view) for view, scaler in zip(views, self.scalers)
        ]
        self.n = views[0].shape[0]
        self.n_views = len(views)
        return views
