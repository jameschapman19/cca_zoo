from typing import Iterable

import numpy as np
import pytorch_lightning as pl
from sklearn import clone
from tqdm import tqdm


# Import tqdm and deflate_views
from tqdm import tqdm


# Define DeflationMixin class
class DeflationMixin:
    # Define fit method
    def fit(self, views: Iterable[np.ndarray], y=None, **kwargs):
        views = self._validate_data(views)
        # Convert views to list if tuple
        if isinstance(views, tuple):
            views = list(views)
        # Initialize weights list
        self.weights = [
            np.empty((view.shape[1], self.latent_dimensions)) for view in views
        ]
        # Loop over latent dimensions
        for k in tqdm(
            range(self.latent_dimensions),
            desc="Latent Dimension",
            position=0,
            leave=True,
        ):
            # clone self but with only one latent dimension and _fit
            component_weights = clone(self).set_params(latent_dimensions=1)._fit(views)
            # Append component_weights to weights list
            for i, weight in enumerate(component_weights):
                self.weights[i][:, k] = weight.squeeze()
            # Deflate views using component_weights
            views = deflate_views(views, component_weights)
        # Return self
        return self


def deflate_views(residuals: Iterable[np.ndarray], weights: Iterable[np.ndarray]):
    """Deflate the residuals by CCA deflation.

    Parameters
    ----------
    residuals : Iterable[np.ndarray]
        The current residual data matrices for each view
    weights : Iterable[np.ndarray]
        The current CCA weights for each view

    Returns
    -------
    Iterable[np.ndarray]
        The deflated residual data matrices for each view
    """
    # Deflate the residuals for each view
    return [
        deflate_view(residual, weight) for residual, weight in zip(residuals, weights)
    ]


def deflate_view(residual: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Deflate view residual by CCA deflation.

    Parameters
    ----------
    residual : np.ndarray
        The current residual data matrix for a view
    weights : np.ndarray
        The current CCA weights for a view

    Returns
    -------
    np.ndarray
        The deflated residual data matrix for a view
    """
    # Compute the score vector for a view
    score = residual @ weights

    # Compute the deflation term once and reuse it
    deflation_term = (score @ weights.T) / (weights.T @ weights)

    # Use in-place subtraction to reduce memory consumption
    residual -= deflation_term

    return residual
