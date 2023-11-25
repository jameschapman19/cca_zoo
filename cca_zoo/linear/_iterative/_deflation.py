from typing import Iterable

import numpy as np
from sklearn import clone
from tqdm import tqdm


class _DeflationMixin:
    def fit(self, views: Iterable[np.ndarray], y=None, **kwargs):
        views = self._validate_data(views)
        if isinstance(views, tuple):
            views = list(views)

        self.weights_ = [
            np.empty((view.shape[1], self.latent_dimensions)) for view in views
        ]

        for k in tqdm(
            range(self.latent_dimensions),
            desc="Latent Dimension",
            position=0,
            leave=True,
        ):
            component_weights = clone(self).set_params(latent_dimensions=1)._fit(views)

            for i, weight in enumerate(component_weights):
                self.weights_[i][:, k] = weight.squeeze()

            if k < self.latent_dimensions - 1:
                pls = True  # self._get_tags().get("pls", False)
                views = deflate_views(views, component_weights, pls)

        return self


def deflate_views(
    residuals: Iterable[np.ndarray], weights: Iterable[np.ndarray], pls=False
):
    if pls:
        return [
            deflate_view_pls(residual, weight)
            for residual, weight in zip(residuals, weights)
        ]

    else:
        return [
            deflate_view_cca(residual, weight)
            for residual, weight in zip(residuals, weights)
        ]


def deflate_view_pls(residual: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Generalised deflation for PLS

    This method ensures orthogonal weights_ in the consecutive associative effects in each data modality

    """
    score = residual @ weights
    deflation_term = (score @ weights.T) / (weights.T @ weights)
    residual -= deflation_term
    return residual


def deflate_view_cca(residual: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    PLS Mode-A deflation/ CCA deflation

    This method ensures orthogonal latent variables in the consecutive associative effects in each data modality.

    """
    score = residual @ weights
    residual -= score @ (score.T @ residual) / (score.T @ score)
    return residual
