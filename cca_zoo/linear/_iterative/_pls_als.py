from typing import Union

import numpy as np

from cca_zoo.linear._iterative._base import _BaseIterative
from cca_zoo.linear._iterative._deflation import _DeflationMixin


class PLS_ALS(_DeflationMixin, _BaseIterative):
    def __init__(
        self,
        latent_dimensions: int = 1,
        copy_data=True,
        random_state=None,
        tol=1e-3,
        accept_sparse=None,
        epochs=100,
        initialization: Union[str, callable] = "uniform",
        early_stopping=False,
        verbose=True,
    ):
        super().__init__(
            latent_dimensions=latent_dimensions,
            copy_data=copy_data,
            random_state=random_state,
            tol=tol,
            accept_sparse=accept_sparse,
            epochs=epochs,
            initialization=initialization,
            early_stopping=early_stopping,
            verbose=verbose,
        )

    def _update_weights(self, views: np.ndarray, i: int):
        # Update the weights_ for the current view using PLS
        # Get the scores of all representations
        scores = np.stack(self.transform(views))
        # Create a mask that is True for elements not equal to i along dim i
        mask = np.arange(scores.shape[0]) != i
        # Apply the mask to scores and sum along dim i
        target = np.sum(scores[mask], axis=0)
        # Compute the new weights_ by computing the covariance between the view and the target
        new_weights = np.cov(np.hstack((views[i], target)).T)[:-1, -1]
        # Normalize the new weights_
        new_weights /= np.linalg.norm(new_weights)
        # Return the new weights_
        return new_weights[:, np.newaxis]
