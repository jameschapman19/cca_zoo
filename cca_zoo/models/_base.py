import itertools
from abc import abstractmethod
from typing import Union, Iterable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.base import BaseEstimator, MultiOutputMixin, RegressorMixin
from sklearn.utils.validation import check_random_state, check_is_fitted, FLOAT_DTYPES


class BaseCCA(BaseEstimator, MultiOutputMixin, RegressorMixin):
    """
    Base class for CCA methods.
    """

    def __init__(
        self,
        latent_dims: int = 1,
        copy_data=True,
        accept_sparse=False,
        random_state: Union[int, np.random.RandomState] = None,
    ):
        """
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

        """
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

    def plot_pairwise_correlations(
        self,
        views: Iterable[np.ndarray],
        ax=None,
        figsize=None,
        **kwargs,
    ):
        """
        Plots the pairwise correlations between the views in each dimension

        Parameters
        ----------
        views : list/tuple of numpy arrays or array likes with the same number of rows (samples)
        ax : matplotlib axes object, optional
            If not provided, a new figure will be created.
        figsize : tuple, optional
            The size of the figure to create. If not provided, the default matplotlib figure size will be used.
        kwargs : any additional keyword arguments required by the given model

        Returns
        -------
        ax : matplotlib axes object

        """
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        corrs = self.pairwise_correlations(views, **kwargs)
        for i in range(corrs.shape[-1]):
            sns.heatmap(
                corrs[:, :, i],
                annot=True,
                vmin=-1,
                vmax=1,
                center=0,
                cmap="RdBu_r",
                ax=ax,
                **kwargs,
            )
            ax.set_title(f"Dimension {i + 1}")
        return ax

    def plot_pairwise_scatter(
        self, views: Iterable[np.ndarray], ax=None, figsize=None, **kwargs
    ):
        """
        Plots the pairwise scatterplots between the views in each dimension

        Parameters
        ----------
        views : list/tuple of numpy arrays or array likes with the same number of rows (samples)
        ax : matplotlib axes object, optional
            If not provided, a new figure will be created.
        figsize : tuple, optional
            The size of the figure to create. If not provided, the default matplotlib figure size will be used.
        kwargs : any additional keyword arguments required by the given model

        Returns
        -------
        ax : matplotlib axes object

        """
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        transformed_views = self.transform(views, **kwargs)
        for i in range(len(transformed_views)):
            for j in range(i + 1, len(transformed_views)):
                ax.scatter(transformed_views[i], transformed_views[j])
                ax.set_title(f"Dimension {i + 1} vs Dimension {j + 1}")
        return ax

    def plot_each_view_tsne(
        self, views: Iterable[np.ndarray], ax=None, figsize=None, **kwargs
    ):
        """
        Plots the pairwise scatterplots between the views in each dimension

        Parameters
        ----------
        views : list/tuple of numpy arrays or array likes with the same number of rows (samples)
        ax : matplotlib axes object, optional
            If not provided, a new figure will be created.
        figsize : tuple, optional
            The size of the figure to create. If not provided, the default matplotlib figure size will be used.
        kwargs : any additional keyword arguments required by the given model

        Returns
        -------
        ax : matplotlib axes object

        """
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        transformed_views = self.transform(views, **kwargs)
        for i in range(len(transformed_views)):
            ax.scatter(transformed_views[i][:, 0], transformed_views[i][:, 1])
            ax.set_title(f"Dimension {i + 1}")
        return ax


class PLSMixin:
    def total_variance_(self, views: Iterable[np.ndarray], **kwargs) -> np.ndarray:
        """
        Returns the total variance for each view

        Parameters
        ----------
        views : list/tuple of numpy arrays or array likes with the same number of rows (samples)
        kwargs : any additional keyword arguments required by the given model

        Returns
        -------
        variance : numpy array of shape (n_views, latent_dims)

        """
        # Calculate total variance in each view by SVD
        n = views[0].shape[0]
        variance = np.array(
            [np.sum(np.linalg.svd(view)[1] ** 2, keepdims=True) / n for view in views]
        )
        return variance

    def total_covariance_(self, views: Iterable[np.ndarray], **kwargs) -> float:
        """
        Returns the total covariance

        Parameters
        ----------
        views : list/tuple of numpy arrays or array likes with the same number of rows (samples)
        kwargs : any additional keyword arguments required by the given model

        Returns
        -------
        covariance : float

        """
        # Calculate total covariance by SVD
        n = views[0].shape[0]
        # if n<p calculate views[0]@views[0].T@views[1]@views[1].T else calculate views[0].T@views[1]
        if n < min(views[0].shape[1], views[1].shape[1]):
            covariance = (
                np.sum(np.linalg.svd(views[0] @ views[0].T @ views[1] @ views[1].T)[1])
                / n
            )
        else:
            covariance = np.sum(np.linalg.svd(views[0].T @ views[1])[1]) / n
        return covariance

    def explained_variance_(self, views: Iterable[np.ndarray], **kwargs) -> np.ndarray:
        """
        Returns the total variance for each view

        Parameters
        ----------
        views : list/tuple of numpy arrays or array likes with the same number of rows (samples)
        kwargs : any additional keyword arguments required by the given model

        Returns
        -------
        variance : numpy array of shape (n_views, latent_dims)

        """
        transformed_views = self.transform(views, **kwargs)
        variance = np.array([np.var(view, axis=0) for view in transformed_views])
        return variance

    def explained_covariance_(
        self, views: Iterable[np.ndarray], **kwargs
    ) -> np.ndarray:
        """
        Returns the total covariance for each view

        Parameters
        ----------
        views : list/tuple of numpy arrays or array likes with the same number of rows (samples)
        kwargs : any additional keyword arguments required by the given model

        Returns
        -------
        covariance : numpy array of shape (n_views, latent_dims)

        """
        transformed_views = self.transform(views, **kwargs)
        covariance = np.cov(*transformed_views, rowvar=False)[
            : self.latent_dims, self.latent_dims :
        ]
        return covariance

    def explained_covariance_ratio(
        self, views: Iterable[np.ndarray], **kwargs
    ) -> np.ndarray:
        """
        Returns the explained covariance ratio for each view

        Parameters
        ----------
        views : list/tuple of numpy arrays or array likes with the same number of rows (samples)
        kwargs : any additional keyword arguments required by the given model

        Returns
        -------
        explained_covariance_ratio : numpy array of shape (n_views, latent_dims)

        """
        explained_covariance = self.explained_covariance_(views, **kwargs)
        covariance = self.total_covariance_(views, **kwargs)
        explained_covariance_ratio = explained_covariance / covariance
        return explained_covariance_ratio

    def explained_variance_ratio_(
        self, views: Iterable[np.ndarray], **kwargs
    ) -> np.ndarray:
        """
        Returns the explained variance ratio for each view

        Parameters
        ----------
        views : list/tuple of numpy arrays or array likes with the same number of rows (samples)
        kwargs : any additional keyword arguments required by the given model

        Returns
        -------
        explained_variance_ratio : numpy array of shape (n_views, latent_dims)

        """
        component_variance = self.explained_variance_(views, **kwargs)
        variance = self.total_variance_(views, **kwargs)
        explained_variance_ratio = component_variance / variance
        return explained_variance_ratio

    def explained_variance_cumulative(
        self, views: Iterable[np.ndarray], **kwargs
    ) -> np.ndarray:
        """
        Returns the cumulative explained variance for each view

        Parameters
        ----------
        views : list/tuple of numpy arrays or array likes with the same number of rows (samples)
        kwargs : any additional keyword arguments required by the given model

        Returns
        -------
        explained_variance_cumulative : numpy array of shape (n_views, latent_dims)

        """
        explained_variance_ratio = self.explained_variance_ratio_(views, **kwargs)
        explained_variance_cumulative = np.cumsum(explained_variance_ratio, axis=1)
        return explained_variance_cumulative

    def plot_explained_variance(
        self, views: Iterable[np.ndarray], ax=None, figsize=None, **kwargs
    ):
        """
        Plots the explained variance for each dimension

        Parameters
        ----------
        views : list/tuple of numpy arrays or array likes with the same number of rows (samples)
        ax : matplotlib axes object, optional
              If not provided, a new figure will be created.
        figsize : tuple, optional
              The size of the figure to create. If not provided, the default matplotlib figure size will be used.
        kwargs : any additional keyword arguments required by the given model

        Returns
        -------
        ax : matplotlib axes object

        """
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        variances = self.explained_variance_(views, **kwargs)
        ax.plot(np.arange(1, variances.shape[-1] + 1), variances.mean(axis=(0, 1)))
        ax.set_xlabel("Dimension")
        ax.set_ylabel("Explained Variance")
        return ax
