from typing import Iterable

import numpy as np


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

            def reduce_dims(x):
                U, S, _ = np.linalg.svd(x, full_matrices=False)
                return U @ np.diag(S)

            covariance = np.sum(
                np.linalg.svd(reduce_dims(views[0]).T @ reduce_dims(views[1]))[1]
            ) / (n - 1)
        else:
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
        covariance = np.diag(
            np.cov(*transformed_views, rowvar=False)[
                : self.latent_dims, self.latent_dims :
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
        explained_covariance_ratio_ : numpy array of shape (n_views, latent_dims)

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
        explained_variance_cumulative_ : numpy array of shape (n_views, latent_dims)

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
        explained_covariance_cumulative_ : numpy array of shape (n_views, latent_dims)

        """
        explained_covariance_ratio = self.explained_covariance_ratio_(views, **kwargs)
        explained_covariance_cumulative = np.cumsum(explained_covariance_ratio)
        return explained_covariance_cumulative
