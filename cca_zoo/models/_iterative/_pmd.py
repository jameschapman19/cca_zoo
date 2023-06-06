import warnings

import numpy as np

from cca_zoo.models._base import PLSMixin
from cca_zoo.models._iterative._base import BaseDeflation, BaseLoop
from cca_zoo.models._search import _delta_search
from cca_zoo.utils import _check_converged_weights, _process_parameter


class SCCA_PMD(BaseDeflation, PLSMixin):
    def __init__(
        self,
        latent_dims=1,
        copy_data=True,
        random_state=None,
        deflation="cca",
        initialization="pls",
        tol=1e-3,
        positive=False,
        tau=None,
    ):
        super().__init__(
            latent_dims=latent_dims,
            copy_data=copy_data,
            random_state=random_state,
            deflation=deflation,
            initialization=initialization,
            tol=tol,
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

    def _get_module(self, weights=None, k=None):
        return PMDLoop(
            weights=weights,
            k=k,
            tau=self.tau,
            tol=self.tol,
        )

    def _more_tags(self):
        return {"multiview": True}


class PMDLoop(BaseLoop):
    def __init__(self, weights, k=None, tau=None, tol=1e-3):
        super().__init__(weights=weights, k=k)
        self.tau = tau
        self.tol = tol
        shape_sqrts = [np.sqrt(weight.shape[0]) for weight in self.weights]
        self.t = [max(1, x * y) for x, y in zip(self.tau, shape_sqrts)]

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
            self.weights[view_index] = _delta_search(
                self.weights[view_index],
                self.t[view_index],
                tol=self.tol,
            )
            _check_converged_weights(self.weights[view_index], view_index)
