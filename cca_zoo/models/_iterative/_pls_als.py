from typing import Union

import numpy as np

from cca_zoo.models._base import PLSMixin
from cca_zoo.models._iterative._base import BaseDeflation, BaseLoop


class PLS_ALS(BaseDeflation, PLSMixin):
    def __init__(
        self,
        latent_dims: int = 1,
        copy_data=True,
        random_state=None,
        tol=1e-3,
        deflation="pls",
        accept_sparse=None,
        batch_size=None,
        dataloader_kwargs=None,
        epochs=100,
        val_split=None,
        learning_rate=1,
        initialization: Union[str, callable] = "random",
        callbacks=None,
        trainer_kwargs=None,
    ):
        super().__init__(
            latent_dims,
            copy_data,
            random_state,
            tol,
            deflation=deflation,
            accept_sparse=accept_sparse,
            batch_size=batch_size,
            dataloader_kwargs=dataloader_kwargs,
            epochs=epochs,
            val_split=val_split,
            learning_rate=learning_rate,
            initialization=initialization,
            callbacks=callbacks,
            trainer_kwargs=trainer_kwargs,
        )

    def _get_module(self, weights=None, k=None):
        return PlsAlsLoop(
            weights=weights,
            k=k,
        )

    def _more_tags(self):
        return {"multiview": True}


class PlsAlsLoop(BaseLoop):
    def training_step(self, batch, batch_idx):
        scores = np.stack(self(batch["views"]))
        # Update each view using loop update function
        for view_index, view in enumerate(batch["views"]):
            # create a mask that is True for elements not equal to k along dim k
            mask = np.arange(scores.shape[0]) != view_index
            # apply the mask to scores and sum along dim k
            target = np.sum(scores[mask], axis=0)
            self.weights[view_index] = np.cov(
                np.hstack((batch["views"][view_index], target[:, np.newaxis])).T
            )[:-1, -1]
            self.weights[view_index] /= np.linalg.norm(self.weights[view_index])
