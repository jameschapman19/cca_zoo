import warnings
from typing import Iterable

import numpy as np

from cca_zoo.models._multiview._mcca import MCCA
from ..utils import _process_parameter


class GRCCA(MCCA):
    """
    Grouped Regularized Canonical Correlation Analysis

    Parameters
    ----------
    latent_dims : int, default=1
        Number of latent dimensions to use
    scale : bool, default=True
        Whether to scale the data to unit variance
    centre : bool, default=True
        Whether to centre the data
    copy_data : bool, default=True
        Whether to copy the data
    random_state : int, default=None
        Random state for initialisation
    eps : float, default=1e-3
        Tolerance for convergence
    c : float, default=0
        Regularization parameter for the group means
    mu : float, default=0
        Regularization parameter for the group sizes


    References
    ----------
    Tuzhilina, Elena, Leonardo Tozzi, and Trevor Hastie. "Canonical correlation analysis in high dimensions with structured regularization." Statistical Modelling (2021): 1471082X211041033.
    """

    def __init__(
        self,
        latent_dims: int = 1,
        scale: bool = True,
        centre=True,
        copy_data=True,
        random_state=None,
        eps=1e-3,
        c: float = 0,
        mu: float = 0,
    ):
        super().__init__(
            latent_dims=latent_dims,
            scale=scale,
            centre=centre,
            copy_data=copy_data,
            random_state=random_state,
            eps=eps,
            c=c,
        )
        self.mu = mu

    def _check_params(self):
        self.mu = _process_parameter("mu", self.mu, 0, self.n_views)
        self.c = _process_parameter("c", self.c, 0, self.n_views)

    def fit(self, views: Iterable[np.ndarray], y=None, feature_groups=None, **kwargs):
        """
        Parameters
        ----------
        views : list/tuple of numpy arrays or array likes with the same number of rows (samples)
        y : None
        feature_groups : list/tuple of integer numpy arrays or array likes with dimensions (,view shape)
        kwargs: any additional keyword arguments required by the given model

        """
        if feature_groups is None:
            warnings.warn(f"No feature groups provided, using all features")
            feature_groups = [np.ones(view.shape[1], dtype=int) for view in views]
        for feature_group in feature_groups:
            assert np.issubdtype(
                feature_group.dtype, np.integer
            ), "feature groups must be integers"
        views = self._validate_inputs(views)
        self._check_params()
        views, idxs = self._preprocess(views, feature_groups)
        C, D = self._setup_evp(views, **kwargs)
        eigvals, eigvecs = self._solve_evp(C, D)
        self._weights(eigvals, eigvecs, views)
        self._transform_weights(views, feature_groups)
        return self

    def _preprocess(self, views, feature_groups):
        views, idxs = list(
            zip(
                *[
                    self._process_view(view, group, mu, c)
                    for view, group, mu, c in zip(
                        views, feature_groups, self.mu, self.c
                    )
                ]
            )
        )
        return views, idxs

    @staticmethod
    def _process_view(view, group, mu, c):
        if c > 0:
            ids, unique_inverse, unique_counts, group_means = _group_mean(view, group)
            if mu == 0:
                mu = 1
                idx = view.shape[1] - 1
            else:
                idx = view.shape[1] + group_means.shape[1] - 1
            view_1 = (view - group_means[:, unique_inverse]) / c
            view_2 = group_means / np.sqrt(mu / unique_counts)
            return np.hstack((view_1, view_2)), idx
        else:
            return view, view.shape[1] - 1

    def _transform_weights(self, views, groups):
        for i, (view, group) in enumerate(zip(views, groups)):
            if self.c[i] > 0:
                weights_1 = self.weights[i][: len(group)]
                weights_2 = self.weights[i][len(group) :]
                ids, unique_inverse, unique_counts, group_means = _group_mean(
                    weights_1.T, group
                )
                weights_1 = (weights_1 - group_means[:, unique_inverse].T) / self.c[i]
                if self.mu[i] == 0:
                    mu = 1
                else:
                    mu = self.mu[i]
                weights_2 = weights_2 / np.sqrt(
                    mu * np.expand_dims(unique_counts, axis=1)
                )
                self.weights[i] = weights_1 + weights_2[group]


def _group_mean(view, group):
    ids, unique_inverse, unique_counts = np.unique(
        group, return_inverse=True, return_counts=True
    )
    group_means = np.array([view[:, group == id].mean(axis=1) for id in ids]).T
    return ids, unique_inverse, unique_counts, group_means
