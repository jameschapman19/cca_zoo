import itertools
import warnings
from typing import Union, Iterable

import numpy as np

from cca_zoo._utils._checks import _process_parameter
from cca_zoo._utils._cross_correlation import cross_corrcoef
from cca_zoo.linear._iterative._base import _BaseIterative
from cca_zoo.linear._iterative._deflation import _DeflationMixin
from cca_zoo.linear._pls import PLSMixin
from cca_zoo.linear._search import _delta_search


class SPLS(_DeflationMixin, _BaseIterative, PLSMixin):
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
        tau=None,  # regularization parameter for PMD
        positive=False,
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
        self.positive = positive

    def _check_params(self):
        if self.tau is None:
            warnings.warn(
                "tau parameter not set. Setting to tau=1 i.e. maximum regularisation of l1 norm"
            )
        self.tau = _process_parameter("tau", self.tau, 1, self.n_views_)
        if any(tau < 0 or tau > 1 for tau in self.tau):
            raise ValueError(
                "All regularisation parameters should be between 0 and 1 "
                f"1. tau=[{self.tau}]"
            )
        self.positive = _process_parameter(
            "positive", self.positive, False, self.n_views_
        )

    def _update_weights(self, views: np.ndarray, i: int):
        if not hasattr(self, "t"):
            shape_sqrts = [np.sqrt(weight.shape[0]) for weight in self.weights_]
            self.t = [max(1, x * y) for x, y in zip(self.tau, shape_sqrts)]
        # Update the weights_ for the current view using PMD
        # Get the scores of all representations
        scores = np.stack(self.transform(views))
        # Create a mask that is True for elements not equal to i along dim i
        mask = np.arange(scores.shape[0]) != i
        # Apply the mask to scores and sum along dim i
        target = np.sum(scores[mask], axis=0)
        # Compute the new weights_ by multiplying the view with the target
        new_weights = views[i].T @ target
        if self.positive[i]:
            # If positive is true, set all negative values to 0
            new_weights[new_weights < 0] = 0
        # Apply the delta search function to the new weights_ with the regularization parameter
        new_weights = _delta_search(new_weights, self.t[i], tol=self.tol)
        # Return the new weights_
        return new_weights

    def _objective(self, views: Iterable[np.ndarray]):
        # Compute the objective function value for a given set of representations using SCCA
        # Get the scores of all representations
        transformed_views = self.transform(views)
        all_covs = []
        # Sum all the pairwise covariances except self-covariance
        for x, y in itertools.product(transformed_views, repeat=2):
            all_covs.append(np.diag(cross_corrcoef(x.T, y.T)))
        # the sum of covariances
        return np.sum(all_covs) - np.sum(
            [
                self.tau[i] * np.linalg.norm(self.weights_[i])
                for i in range(len(self.weights_))
            ]
        )

    def _more_tags(self):
        # Indicate that this class is for multiview data
        return {"multiview": True}
