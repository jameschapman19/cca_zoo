from typing import Iterable

import numpy as np
from scipy.linalg import block_diag

from ._mcca import MCCA
from ._rcca import _pca_data
from ..utils import _process_parameter


class PRCCA(MCCA):
    """
    :Citation:

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
        c=0,
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

    def fit(self, views: Iterable[np.ndarray], y=None, idxs=None, **kwargs):
        views = self._validate_inputs(views)
        self._check_params()
        views = self.preprocess(views, idxs)
        views, C, D = self._setup_evp(views, idxs, **kwargs)
        self._solve_evp(views, C, D, **kwargs)
        self.transform_weights(views, idxs)
        return self

    def preprocess(self, views, idxs):
        Us, Ss, self.Vts = _pca_data(*[view[:, :idx] for view, idx in zip(views, idxs)])
        self.betas = [
            np.linalg.pinv(view[:, idxs[i] :]) @ U @ np.diag(S)
            for i, (view, U, S) in enumerate(zip(views, Us, Ss))
        ]
        partials = [
            U @ np.diag(S) - view[:, idxs[i] :] @ beta
            for i, (view, U, S, beta) in enumerate(zip(views, Us, Ss, self.betas))
        ]
        views = [
            np.hstack((partial, view[:, idx:]))
            for partial, view, idx in zip(partials, views, idxs)
        ]
        return views

    def transform_weights(self, views, idxs):
        for i, view in enumerate(views):
            self.weights[i][: idxs[i], :] = (
                self.Vts[i].T @ self.weights[i][: idxs[i], :]
            )
            self.weights[i][idxs[i] :, :] = (
                -self.betas[i] @ self.weights[i][: idxs[i], :]
                + self.weights[i][idxs[i] :]
            )

    def _setup_evp(self, views: Iterable[np.ndarray], idxs, **kwargs):
        n = views[0].shape[0]
        all_views = np.concatenate(views, axis=1)
        C = all_views.T @ all_views / self.n
        # Can regularise by adding to diagonal
        self.c_ = [
            np.append(
                self.c[i] * np.ones(idxs[i] + 1),
                np.zeros(views[i].shape[1] - 1 - idxs[i]),
            )
            for i, view in enumerate(views)
        ]
        D = block_diag(
            *[(m.T @ m) / self.n + np.diag(self.c_[i]) for i, m in enumerate(views)]
        )
        C -= block_diag(*[view.T @ view / self.n for view in views])
        D_smallest_eig = min(0, np.linalg.eigvalsh(D).min()) - self.eps
        D = D - D_smallest_eig * np.eye(D.shape[0])
        self.splits = np.cumsum([0] + [view.shape[1] for view in views])
        return views, C, D


class GRCCA(PRCCA):
    """
    :Citation:

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

    def fit(self, views: Iterable[np.ndarray], y=None, groups=None, **kwargs):
        views = self._validate_inputs(views)
        self._check_params()
        views, idxs = self.preprocess(views, groups)
        self.weights = (
            PRCCA(latent_dims=self.latent_dims, scale=False, centre=False, c=1)
            .fit(views, idxs=idxs, **kwargs)
            .weights
        )
        self.transform_weights(views, groups)
        return self

    def preprocess(self, views, groups):
        views, idxs = list(
            zip(
                *[
                    self._process_view(view, group, mu, c)
                    for view, group, mu, c in zip(views, groups, self.mu, self.c)
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

    def transform_weights(self, views, groups):
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
