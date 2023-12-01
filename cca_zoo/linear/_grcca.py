import warnings
from typing import Iterable

import numpy as np

from cca_zoo._utils._checks import _process_parameter
from cca_zoo.linear._mcca import MCCA


class GRCCA(MCCA):
    """
    Grouped Regularized Canonical Correlation Analysis

    Parameters
    ----------
    latent_dimensions: int, default=1
        Number of latent dimensions to use
    copy_data: bool, default=True
        Whether to copy the data
    random_state: int, default=None
        Random state for initialisation
    eps: float, default=1e-3
        Tolerance for convergence
    c: float, default=0
        Regularization parameter for the group means
    mu: float, default=0
        Regularization parameter for the group sizes


    References
    ----------
    Tuzhilina, Elena, Leonardo Tozzi, and Trevor Hastie. "Canonical correlation analysis in high dimensions with structured regularization." Statistical Modelling (2021): 1471082X211041033.
    """

    def __init__(
        self,
        latent_dimensions: int = 1,
        copy_data=True,
        random_state=None,
        eps=1e-3,
        c: float = 0,
        mu: float = 0,
    ):
        super().__init__(
            latent_dimensions=latent_dimensions,
            copy_data=copy_data,
            random_state=random_state,
            eps=eps,
            c=c,
            pca=False,
        )
        self.mu = mu

    def _check_params(self):
        self.mu = _process_parameter("mu", self.mu, 0, self.n_views_)
        self.c = _process_parameter("c", self.c, 0, self.n_views_)

    def fit(self, views: Iterable[np.ndarray], y=None, feature_groups=None, **kwargs):
        return super().fit(views, y=y, feature_groups=feature_groups, **kwargs)

    def _weights(self, eigvals, eigvecs, views, feature_groups=None, **kwargs):
        # Loop through c and add group means to splits if c > 0
        self.splits = [
            n_features + n_groups if c > 0 else n_features
            for n_features, n_groups, c in zip(
                self.n_features_in_, self.n_groups_, self.c
            )
        ]

        # Add zero at the beginning and compute cumulative sum of splits
        self.splits = np.insert(np.cumsum(self.splits), 0, 0)

        # Slice eigenvectors according to splits
        self.weights_ = [
            eigvecs[split:next_split]
            for split, next_split in zip(self.splits[:-1], self.splits[1:])
        ]

        # Adjust weights_ for each view based on group means and mu parameters
        for i, view in enumerate(views):
            if self.c[i] > 0:
                weights_1 = self.weights_[i][: -self.n_groups_[i]]
                weights_2 = self.weights_[i][-self.n_groups_[i] :]
                ids, unique_inverse, unique_counts, group_means = self._group_mean(
                    weights_1.T, feature_groups[i]
                )
                weights_1 = (weights_1 - group_means[:, unique_inverse].T) / self.c[i]
                mu = 1 if self.mu[i] == 0 else self.mu[i]
                weights_2 = weights_2 / np.sqrt(
                    mu * np.expand_dims(unique_counts, axis=1)
                )
                self.weights_[i] = weights_1 + weights_2[unique_inverse]

    def _process_data(self, views, feature_groups=None, **kwargs):
        # Use all features if no feature groups are provided
        if feature_groups is None:
            warnings.warn("No feature groups provided, using all features")
            feature_groups = [np.ones(view.shape[1], dtype=int) for view in views]

        # Check that feature groups are integers
        for feature_group in feature_groups:
            assert np.issubdtype(
                feature_group.dtype, np.integer
            ), "feature groups must be integers"

        # Number of unique groups in each view
        self.n_groups_ = [np.unique(group).shape[0] for group in feature_groups]
        # Process each view and return a list of processed representations and indices
        return [
            self._process_view(view, group, mu, c)
            for view, group, mu, c in zip(views, feature_groups, self.mu, self.c)
        ]

    def _process_view(self, view, group, mu, c):
        """
        Process a single view by subtracting group means and adding them as new features.

        Parameters
        ----------
        view: numpy array or array like with shape (n_samples, n_features)
            The view to be processed.

        group: numpy array or array like with shape (n_features,)
            The feature group labels for the view.

        mu: float
            The regularization parameter for the group means.

        c: float
            The regularization parameter for the view features.

        Returns
        -------
        view: numpy array with shape (n_samples, n_features + n_groups)
            The processed view with group means added as new features.
        """
        if c > 0:
            (
                ids,
                unique_inverse,
                unique_counts,
                group_means,
            ) = self._group_mean(view, group)
            mu = 1 if mu == 0 else mu
            view_1 = (view - group_means[:, unique_inverse]) / c
            view_2 = group_means / np.sqrt(mu / unique_counts)
            return np.hstack((view_1, view_2))
        else:
            return view

    def _more_tags(self):
        return {"multiview": True}

    @staticmethod
    def _group_mean(view, group):
        """
        Compute the mean of each feature group in a view.

        Parameters
        ----------
        view: numpy array or array like with shape (n_samples, n_features)
            The view to compute the group means from.

        group: numpy array or array like with shape (n_features,)
            The feature group labels for the view.

        Returns
        -------
        ids: numpy array with shape (n_groups,)
            The unique feature group ids.

        unique_inverse: numpy array with shape (n_features,)
            The indices to reconstruct the original group array from the unique ids.

        unique_counts: numpy array with shape (n_groups,)
            The number of occurrences of each unique id in the group array.

        group_means: numpy array with shape (n_samples, n_groups)
            The mean of each feature group in the view.
        """
        ids, unique_inverse, unique_counts = np.unique(
            group, return_inverse=True, return_counts=True
        )
        # Use axis argument to compute mean along columns for each group
        group_means = np.array([view[:, group == id].mean(axis=1) for id in ids]).T
        return ids, unique_inverse, unique_counts, group_means
