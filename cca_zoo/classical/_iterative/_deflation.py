from abc import ABC
from typing import Iterable
import pytorch_lightning as pl
from tqdm import tqdm
import numpy as np
from cca_zoo.classical._iterative._base import BaseIterative


class BaseDeflation(BaseIterative, ABC):
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
                views[i] = self._deflate(view, weight)
            # if loop has tracked the objective, return the objective
            if hasattr(loop, "epoch_objective"):
                self.objective = loop.epoch_objective
        return self.weights

    def _deflate(self, residual: np.ndarray, weights: np.ndarray) -> np.ndarray:
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
        if self.deflation == "cca":
            return residual - np.outer(score, score) @ residual / np.dot(score, score)
        elif self.deflation == "pls":
            return residual - np.outer(score, weights)
        else:
            raise ValueError(
                f"Invalid deflation method: {self.deflation}. "
                f"Must be one of ['cca', 'pls']."
            )
