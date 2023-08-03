from typing import Iterable

import numpy as np
import pytorch_lightning as pl
from tqdm import tqdm


class DeflationMixin:
    def _fit(self, views: Iterable[np.ndarray]):
        # if views is a tuple then convert to a list
        if isinstance(views, tuple):
            views = list(views)
        # tqdm for each latent dimension
        for k in tqdm(
            range(self.latent_dimensions), desc="Latent Dimension", leave=False
        ):
            train_dataloader, val_dataloader = self.get_dataloader(views)
            loop = self._get_module(weights=self.weights, k=k)
            # make a trainer
            trainer = pl.Trainer(
                max_epochs=self.epochs, callbacks=self.callbacks, **self.trainer_kwargs
            )
            trainer.fit(loop, train_dataloader, val_dataloader)
            # return the weights from the module. They will need to be changed from torch tensors to numpy arrays
            weights = loop.weights
            for i, (view, weight) in enumerate(zip(views, weights)):
                self.weights[i][:, k] = weight
            views = deflate_views(views, weights)
            # if loop has tracked the objective, return the objective
            if hasattr(loop, "epoch_objective"):
                self.objective = loop.epoch_objective
        return self.weights


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

    Raises
    ------
    ValueError
        If deflation method is not one of ["cca", "pls"]
    """
    # Compute the score vector for a view
    score = residual @ weights

    # Deflate the residual by different methods based on the deflation attribute
    return residual - residual @ weights @ weights.T / (weights.T @ weights)
