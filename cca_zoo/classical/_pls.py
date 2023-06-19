from cca_zoo.classical import rCCA, MCCA
from typing import Iterable

import numpy as np


def reduce_dims(x):
    U, S, _ = np.linalg.svd(x, full_matrices=False)
    return U @ np.diag(S)


class PLSMixin:
    def total_correlation_(self, views: Iterable[np.ndarray], **kwargs) -> np.ndarray:
        """
        Returns the total correlation for each view

        Parameters
        ----------
        views : list/tuple of numpy arrays or array likes with the same number of rows (samples)
        kwargs : any additional keyword arguments required by the given model

        Returns
        -------
        correlation : numpy array of shape (n_views, latent_dimensions)

        """
        # Calculate total correlation in each view by SVD
        n = views[0].shape[0]
        views = list(views)
        for i, view in enumerate(views):
            if n < view.shape[1]:
                views[i] = reduce_dims(view)
        correlation = (
            MCCA(latent_dimensions=min([view.shape[1] for view in views]))
            .fit(views)
            .score(views)
            .sum()
        )
        return correlation

    def total_variance_(self, views: Iterable[np.ndarray], **kwargs) -> np.ndarray:
        """
        Returns the total variance for each view

        Parameters
        ----------
        views : list/tuple of numpy arrays or array likes with the same number of rows (samples)
        kwargs : any additional keyword arguments required by the given model

        Returns
        -------
        variance : numpy array of shape (n_views, latent_dimensions)

        """
        views = list(views)
        # Calculate total variance in each view by SVD
        n = views[0].shape[0]
        for i, view in enumerate(views):
            if n < view.shape[1]:
                views[i] = reduce_dims(view)
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
        views = list(views)
        # Calculate total covariance by SVD
        n = views[0].shape[0]
        for i, view in enumerate(views):
            if n < view.shape[1]:
                views[i] = reduce_dims(view)
        covariance = np.sum(np.linalg.svd(views[0].T @ views[1])[1]) / (n - 1)
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
        variance : numpy array of shape (n_views, latent_dimensions)

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
        covariance : numpy array of shape (n_views, latent_dimensions)

        """
        transformed_views = self.transform(views, **kwargs)
        covariance = np.diag(
            np.cov(*transformed_views, rowvar=False)[
                : self.latent_dimensions, self.latent_dimensions :
            ]
        )
        return covariance

    def explained_covariance_ratio_(
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
        explained_covariance_ratio_ : numpy array of shape (n_views, latent_dimensions)

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
        explained_variance_ratio : numpy array of shape (n_views, latent_dimensions)

        """
        component_variance = self.explained_variance_(views, **kwargs)
        variance = self.total_variance_(views, **kwargs)
        explained_variance_ratio = component_variance / variance
        return explained_variance_ratio

    def explained_variance_cumulative_(
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
        explained_variance_cumulative_ : numpy array of shape (n_views, latent_dimensions)

        """
        explained_variance_ratio = self.explained_variance_ratio_(views, **kwargs)
        explained_variance_cumulative = np.cumsum(explained_variance_ratio, axis=1)
        return explained_variance_cumulative

    def explained_covariance_cumulative_(self, views: Iterable[np.ndarray], **kwargs):
        """
        Returns the cumulative explained covariance for each view

        Parameters
        ----------
        views : list/tuple of numpy arrays or array likes with the same number of rows (samples)
        kwargs : any additional keyword arguments required by the given model

        Returns
        -------
        explained_covariance_cumulative_ : numpy array of shape (n_views, latent_dimensions)

        """
        explained_covariance_ratio = self.explained_covariance_ratio_(views, **kwargs)
        explained_covariance_cumulative = np.cumsum(explained_covariance_ratio)
        return explained_covariance_cumulative

    def total_variance_captured(self, views: Iterable[np.ndarray], **kwargs):
        """
        Returns the total variance captured by the latent space

        Parameters
        ----------
        views : list/tuple of numpy arrays or array likes with the same number of rows (samples)
        kwargs : any additional keyword arguments required by the given model

        Returns
        -------
        total_variance_captured : float

        """
        transformed_views = self.transform(views, **kwargs)
        total_variance_captured = self.total_variance_(transformed_views)
        return total_variance_captured

    def total_covariance_captured(self, views: Iterable[np.ndarray], **kwargs):
        """
        Returns the total covariance captured by the latent space

        Parameters
        ----------
        views : list/tuple of numpy arrays or array likes with the same number of rows (samples)
        kwargs : any additional keyword arguments required by the given model

        Returns
        -------
        total_covariance_captured : float

        """
        transformed_views = self.transform(views, **kwargs)
        total_covariance_captured = self.total_covariance_(transformed_views)
        return total_covariance_captured

    def total_correlation_captured(self, views: Iterable[np.ndarray], **kwargs):
        """
        Returns the total correlation captured by the latent space

        Parameters
        ----------
        views : list/tuple of numpy arrays or array likes with the same number of rows (samples)
        kwargs : any additional keyword arguments required by the given model

        Returns
        -------
        total_correlation_captured : float

        """
        transformed_views = self.transform(views, **kwargs)
        total_correlation_captured = self.total_correlation_(transformed_views)
        return total_correlation_captured


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
    latent_dimensions : int, optional
        Number of latent dimensions to use, by default 1
    copy_data : bool, optional
        Whether to copy the data, by default True
    random_state : int, optional
        Random state, by default None
    """

    def __init__(
        self,
        latent_dimensions: int = 1,
        copy_data=True,
        random_state=None,
    ):
        # Call the parent class constructor with c=1 to enable maximal regularization
        super().__init__(
            latent_dimensions=latent_dimensions,
            copy_data=copy_data,
            c=1,
            random_state=random_state,
        )


class MPLS(MCCA, PLSMixin):
    r"""
    A class used to fit a mutiview PLS model. This model finds the linear projections of two views that maximize their covariance.

    Implements PLS by inheriting regularised CCA with maximal regularisation. This is equivalent to solving the following optimization problem:

    Parameters
    ----------
    latent_dimensions : int, optional
        Number of latent dimensions to use, by default 1
    copy_data : bool, optional
        Whether to copy the data, by default True
    random_state : int, optional
        Random state, by default None
    """

    def __init__(
        self,
        latent_dimensions: int = 1,
        copy_data=True,
        random_state=None,
    ):
        # Call the parent class constructor with c=1 to enable maximal regularization
        super().__init__(
            latent_dimensions=latent_dimensions,
            copy_data=copy_data,
            c=1,
            random_state=random_state,
        )
