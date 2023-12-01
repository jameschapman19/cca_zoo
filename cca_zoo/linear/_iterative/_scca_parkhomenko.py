from typing import Union, Iterable

import numpy as np

from cca_zoo._utils._checks import _process_parameter
from cca_zoo.linear._iterative._base import _BaseIterative
from cca_zoo.linear._iterative._deflation import _DeflationMixin


class SCCA_Parkhomenko(_DeflationMixin, _BaseIterative):
    def __init__(
        self,
        latent_dimensions: int = 1,
        copy_data=True,
        random_state=None,
        tol=1e-3,
        accept_sparse=None,
        epochs=100,
        initialization: Union[str, callable] = "pls",
        early_stopping=False,
        verbose=True,
        tau=None,  # regularization parameter for Parkhomenko
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
        self.tau = tau

    def _check_params(self):
        self.tau = _process_parameter("tau", self.tau, 0.0001, self.n_views_)
        if any(tau <= 0 for tau in self.tau):
            raise (
                "All regularisation parameters should be above 0. " f"tau=[{self.tau}]"
            )

    def _update_weights(self, views: Iterable[np.ndarray], i: int):
        # Update the weights_ for the current view using Parkhomenko
        # Get the scores of all representations
        scores = np.stack(self.transform(views))
        # Create a mask that is True for elements not equal to i along dim i
        mask = np.arange(scores.shape[0]) != i
        # Apply the mask to scores and sum along dim i
        target = np.sum(scores[mask], axis=0)
        # Compute the new weights_ by multiplying the view with the target and dividing by the norm of the new weights_
        new_weights = views[i].T @ target / np.linalg.norm(views[i].T @ target)
        # Apply soft thresholding to the new weights_ with optimal delta
        new_weights = np.clip(new_weights - self.tau[i] / 2, 0, None) - np.clip(
            -new_weights - self.tau[i] / 2, 0, None
        )
        # Return the new weights_
        return new_weights
