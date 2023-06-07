import itertools
import warnings

import numpy as np
import torch

from cca_zoo.models._iterative._base import BaseDeflation, BaseLoop
from cca_zoo.models._plsmixin import PLSMixin
from cca_zoo.models._search import _delta_search
from cca_zoo.utils import _check_converged_weights, _process_parameter


class SCCA_PMD(BaseDeflation, PLSMixin):
    r"""
    A class used to fit a sparse CCA model by penalized matrix decomposition (PMD).

    This model finds the linear projections of two views that maximize their correlation while enforcing sparsity constraints on the projection vectors.

    The objective function of SCCA-PMD is:

    .. math::

        w_{opt}=\underset{w}{\mathrm{argmax}}\{ w_1^TX_1^TX_2w_2  \}\\

        \text{subject to:}

        \|w_i\|_2^2=1

        \|w_i\|_1\leq \tau_i

    where :math:`\tau_i` are the sparsity parameters for each view.

    The algorithm alternates between updating the weights for each view and applying a soft thresholding operator to enforce the sparsity constraints.

    Parameters
    ----------
    latent_dims : int, optional
        Number of latent dimensions to use, by default 1
    copy_data : bool, optional
        Whether to copy the data, by default True
    random_state : int, optional
        Random seed for reproducibility, by default None
    epochs : int, optional
        Number of iterations to run the algorithm, by default 100
    deflation : str, optional
        Deflation scheme to use, by default "cca"
    initialization : str, optional
        Initialization scheme to use, by default "pls"
    tol : float, optional
        Tolerance for convergence, by default 1e-3
    positive : Union[Iterable[bool], bool], optional
        Whether to enforce positivity constraints on the weights, by default False
    tau : Union[Iterable[float], float], optional
        Sparsity parameter or list of parameters for each view, by default None. If None, it will be set to 1 for each view.
    convergence_checking : str, optional
        Convergence scheme to use, by default None
    track : Union[Iterable[str], str], optional
        List of metrics to track during training, by default None
    verbose : bool, optional
        Whether to print progress, by default False
    """

    def __init__(
        self,
        latent_dims=1,
        copy_data=True,
        random_state=None,
        epochs=100,
        deflation="cca",
        initialization="pls",
        tol=1e-3,
        positive=False,
        tau=None,
        convergence_checking=None,
        track=None,
        verbose=False,
    ):
        super().__init__(
            latent_dims=latent_dims,
            copy_data=copy_data,
            random_state=random_state,
            epochs=epochs,
            deflation=deflation,
            initialization=initialization,
            tol=tol,
            convergence_checking=convergence_checking,
            patience=0,
            track=track,
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

    def _get_module(self, weights=None, k=None):
        return PMDLoop(
            weights=weights,
            k=k,
            tau=self.tau,
            tol=self.tol,
            tracking=self.track,
            convergence_checking=self.convergence_checking,
        )

    def _more_tags(self):
        return {"multiview": True}


class PMDLoop(BaseLoop):
    def __init__(
        self,
        weights,
        k=None,
        tau=None,
        tol=1e-3,
        tracking=False,
        convergence_checking=False,
    ):
        super().__init__(
            weights=weights,
            k=k,
            tracking=tracking,
            convergence_checking=convergence_checking,
        )
        self.tau = tau
        self.tol = tol
        shape_sqrts = [np.sqrt(weight.shape[0]) for weight in self.weights]
        self.t = [max(1, x * y) for x, y in zip(self.tau, shape_sqrts)]

    def training_step(self, batch, batch_idx):
        scores = np.stack(self(batch["views"]))
        old_weights = self.weights.copy()
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
        # if tracking or convergence_checking is enabled, compute the objective function
        if self.tracking or self.convergence_checking:
            objective = self.objective(batch["views"])
            # check that the maximum change in weights is smaller than the tolerance times the maximum absolute value of the weights
            weights_change = torch.tensor(
                np.max(
                    [
                        np.max(np.abs(old_weights[i] - self.weights[i]))
                        / np.max(np.abs(self.weights[i]))
                        for i in range(len(self.weights))
                    ]
                )
            )
            return {"loss": torch.tensor(objective), "weights_change": weights_change}

    def objective(self, views):
        transformed_views = self(views)
        all_covs = []
        # sum all the pairwise covariances except self covariance
        for i, j in itertools.combinations(range(len(transformed_views)), 2):
            if i != j:
                all_covs.append(
                    np.cov(
                        np.hstack(
                            (
                                transformed_views[i],
                                transformed_views[j],
                            )
                        ).T
                    )
                )
        return np.sum(all_covs)
