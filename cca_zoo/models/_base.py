import itertools
from abc import abstractmethod
from typing import Union, Iterable

import numpy as np
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, MultiOutputMixin, RegressorMixin
from sklearn.utils.sparsefuncs import mean_variance_axis
from sklearn.utils.validation import check_random_state, check_is_fitted

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
        self.n_views = None

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
        raise NotImplementedError

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
        views = _check_views(
            *views, copy=self.copy_data, accept_sparse=self.accept_sparse
        )
        views = self._centre_scale_transform(views)
        transformed_views = []
        for i, (view) in enumerate(views):
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

    def _centre_scale(self, views: Iterable[np.ndarray]):
        """
        Centers and scales the data

        Parameters
        ----------
        views : list/tuple of numpy arrays or array likes with the same number of rows (samples)

        Returns
        -------
        views : list of numpy arrays


        """
        self.view_means = []
        self.view_stds = []
        transformed_views = []
        for view in views:
            if issparse(view):
                view_mean, view_std = mean_variance_axis(view, axis=0)
                self.view_means.append(view_mean)
                self.view_stds.append(view_std)
                view = view - self.view_means[-1]
                view = view / self.view_stds[-1]
            else:
                if self.centre:
                    view_mean = view.mean(axis=0)
                    self.view_means.append(view_mean)
                    view = view - self.view_means[-1]
                if self.scale:
                    view_std = view.std(axis=0, ddof=1)
                    view_std[view_std == 0.0] = 1.0
                    self.view_stds.append(view_std)
                    view = view / self.view_stds[-1]
            transformed_views.append(view)
        return transformed_views

    def _centre_scale_transform(self, views: Iterable[np.ndarray]):
        """
        Centers and scales the data

        Parameters
        ----------
        views : list/tuple of numpy arrays or array likes with the same number of rows (samples)

        Returns
        -------
        views : list of numpy arrays

        """
        if self.centre:
            views = [view - mean for view, mean in zip(views, self.view_means)]
        if self.scale:
            views = [view / std for view, std in zip(views, self.view_stds)]
        return views

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
        views = _check_views(
            *views, copy=self.copy_data, accept_sparse=self.accept_sparse
        )
        views = self._centre_scale(views)
        self.n = views[0].shape[0]
        self.n_views = len(views)
        return views
