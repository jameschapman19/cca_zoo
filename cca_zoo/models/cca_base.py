import itertools
from abc import abstractmethod
from typing import Union, Iterable

import numpy as np
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, MultiOutputMixin, RegressorMixin
from sklearn.utils.sparsefuncs import mean_variance_axis
from sklearn.utils.validation import check_random_state, check_is_fitted

from cca_zoo.utils import check_views, plot_latent_train_test


class _CCA_Base(BaseEstimator, MultiOutputMixin, RegressorMixin):
    """
    A class used as the base for methods in the package. Allows methods to inherit fit_transform, predict_corr,
    and gridsearch_fit when only fit (and transform where it is different to the default) is provided.

    Attributes
    ----------
    weights : list of weights for each view

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
        Constructor for _CCA_Base

        :param latent_dims: number of latent dimensions to fit
        :param scale: normalize variance in each column before fitting
        :param centre: demean data by column before fitting (and before transforming out of sample
        :param copy_data: If True, X will be copied; else, it may be overwritten
        :param accept_sparse: Whether model can take sparse data as input
        :param random_state: Pass for reproducible output across multiple function calls
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
        Fits a given model

        :param views: list/tuple of numpy arrays or array likes with the same number of rows (samples)
        """
        raise NotImplementedError

    def transform(self, views: Iterable[np.ndarray], y=None, **kwargs):
        """
        Transforms data given a fit model

        :param views: numpy arrays with the same number of rows (samples) separated by commas
        :param kwargs: any additional keyword arguments required by the given model
        """
        check_is_fitted(self, attributes=["weights"])
        views = check_views(
            *views, copy=self.copy_data, accept_sparse=self.accept_sparse
        )
        views = self._centre_scale_transform(views)
        transformed_views = []
        for i, (view) in enumerate(views):
            transformed_view = view @ self.weights[i]
            transformed_views.append(transformed_view)
        return transformed_views

    def fit_transform(self, views: Iterable[np.ndarray], y=None, **kwargs):
        """
        Fits and then transforms the training data

        :param views: list/tuple of numpy arrays or array likes with the same number of rows (samples)
        :param kwargs: any additional keyword arguments required by the given model
        """
        return self.fit(views, **kwargs).transform(views)

    def get_loadings(self, views: Iterable[np.ndarray], y=None, **kwargs):
        """
        Returns the model loadings for each view for the given data

        :param views: list/tuple of numpy arrays or array likes with the same number of rows (samples)
        :param kwargs: any additional keyword arguments required by the given model
        """
        transformed_views = self.transform(views, **kwargs)
        views = self._centre_scale_transform(views)
        loadings = [
            view.T @ transformed_view
            for view, transformed_view in zip(views, transformed_views)
        ]
        return loadings

    def correlations(self, views: Iterable[np.ndarray], y=None, **kwargs):
        """
        Predicts the correlation for the given data using the fit model

        :param views: list/tuple of numpy arrays or array likes with the same number of rows (samples)
        :param kwargs: any additional keyword arguments required by the given model
        :return: all_corrs: an array of the pairwise correlations (k,k,self.latent_dims) where k is the number of views
        :rtype: np.ndarray
        """
        transformed_views = self.transform(views, **kwargs)
        all_corrs = []
        for x, y in itertools.product(transformed_views, repeat=2):
            all_corrs.append(
                np.diag(np.corrcoef(x.T, y.T)[: self.latent_dims, self.latent_dims :])
            )
        all_corrs = np.array(all_corrs).reshape(
            (len(views), len(views), self.latent_dims)
        )
        return all_corrs

    def plot_latent(
        self,
        views: Iterable[np.ndarray],
        test_views: Iterable[np.ndarray] = None,
        title="",
    ):
        scores = self.transform(views)
        if test_views is not None:
            test_scores = self.transform(test_views)
        else:
            test_scores = None
        plot_latent_train_test(scores, test_scores, title=title)

    def score(self, views: Iterable[np.ndarray], y=None, **kwargs):
        # by default return the average pairwise correlation in each dimension (for 2 views just the correlation)
        pair_corrs = self.correlations(views, **kwargs)
        # n views
        n_views = pair_corrs.shape[0]
        # sum all the pairwise correlations for each dimension. Subtract the self correlations. Divide by the number of views. Gives average correlation
        dim_corrs = (
            pair_corrs.sum(axis=tuple(range(pair_corrs.ndim - 1))) - n_views
        ) / (n_views ** 2 - n_views)
        return dim_corrs

    def _centre_scale(self, views: Iterable[np.ndarray]):
        """
        Removes the mean of the training data and standardizes for each view and stores mean and standard deviation during training

        :param views: list/tuple of numpy arrays or array likes with the same number of rows (samples)
        :return: train_views: the demeaned numpy arrays to be used to fit the model
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
        Removes the mean and standardizes each view based on the mean and standard deviation of the training data

        :param views: list/tuple of numpy arrays or array likes with the same number of rows (samples)
        """
        if self.centre:
            views = [view - mean for view, mean in zip(views, self.view_means)]
        if self.scale:
            views = [view / std for view, std in zip(views, self.view_stds)]
        return views
