from typing import Iterable

import numpy as np

from cca_zoo.deep.objectives import CCA_EYLoss, PLS_EYLoss
from cca_zoo.deep.data import DoubleNumpyDataset
from cca_zoo.linear._gradient._base import BaseGradientModel


class CCA_EY(BaseGradientModel):
    objective = CCA_EYLoss()

    def _more_tags(self):
        return {"multiview": True, "stochastic": True, "non_deterministic": True}

    def training_step(self, batch, batch_idx):
        representations = self(batch["views"])
        if self.batch_size is None:
            independent_representations = representations
        else:
            independent_views = batch.get("independent_views", None)
            independent_representations = (
                self(independent_views) if independent_views is not None else None
            )
        loss = self.objective.loss(representations, independent_representations)
        # Logging the loss components with "train/" prefix
        for k, v in loss.items():
            self.log(
                f"train/{k}",
                v,
                prog_bar=True,
                on_epoch=True,
                batch_size=batch["views"][0].shape[0],
            )
        return loss["objective"]

    def validation_step(self, batch, batch_idx):
        representations = self(batch["views"])
        if self.batch_size is None:
            independent_representations = representations
        else:
            independent_views = batch.get("independent_views", None)
            independent_representations = (
                self(independent_views) if independent_views is not None else None
            )
        loss = self.objective.loss(representations, independent_representations)
        # Logging the loss components
        for k, v in loss.items():
            self.log(
                f"val/{k}",
                v,
                prog_bar=True,
                on_epoch=True,
                batch_size=batch["views"][0].shape[0],
            )
        return loss["objective"]

    def get_dataset(self, views: Iterable[np.ndarray], validation_views=None):
        dataset = DoubleNumpyDataset(views, batch_size=self.batch_size)
        if validation_views is not None:
            val_dataset = DoubleNumpyDataset(validation_views, self.batch_size)
        else:
            val_dataset = None
        return dataset, val_dataset


class PLS_EY(CCA_EY):
    objective = PLS_EYLoss()

    def training_step(self, batch, batch_idx):
        representations = self(batch["views"])
        loss = self.objective.loss(representations, weights=self.torch_weights)
        # Logging the loss components with "train/" prefix
        for k, v in loss.items():
            self.log(
                f"train/{k}",
                v,
                prog_bar=True,
                on_epoch=True,
                batch_size=batch["views"][0].shape[0],
            )
        return loss["objective"]

    def validation_step(self, batch, batch_idx):
        representations = self(batch["views"])
        loss = self.objective.loss(representations, weights=self.torch_weights)
        # Logging the loss components
        for k, v in loss.items():
            self.log(
                f"val/{k}",
                v,
                prog_bar=True,
                on_epoch=True,
                batch_size=batch["views"][0].shape[0],
            )
        return loss["objective"]