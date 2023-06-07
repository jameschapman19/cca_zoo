import warnings
from typing import Iterable, Union

import numpy as np
import torch
from sklearn.linear_model import ElasticNet, Lasso, Ridge, SGDRegressor

from cca_zoo.models._iterative._base import BaseDeflation, BaseLoop
from cca_zoo.utils import _process_parameter


class ElasticCCA(BaseDeflation):
    r"""
    A class used to fit an elastic CCA model for two or more views of data.

    This model finds the linear projections of multiple views that maximize their pairwise correlations while enforcing L1 and L2 penalties on the projection vectors.

    ElasticCCA uses CCA with an auxiliary variable target i.e. MAXVAR configuration

    .. math::

        w_{opt}, t_{opt}=\underset{w,t}{\mathrm{argmin}}\{\sum_i \|X_iw_i-t\|^2 + \alpha_i\|w_i\|^2_2 + \text{l1_ratio}\|w_i\|_1\}\\

        \text{subject to:}

        t^Tt=n

    Where :math:`\alpha_i` is the constant that multiplies the penalty terms for view i and :math:`\text{l1_ratio}` is the ratio between L1 and L2 penalties.

    Parameters
    ----------
    latent_dims : int, default=1
        Number of latent dimensions to use
    copy_data : bool, default=True
        Whether to copy the data or overwrite it
    random_state : int, default=None
        Random seed for initialization
    deflation : str, default="cca"
        Whether to use CCA or PLS deflation
    initialization : str or callable, default="pls"
        How to initialize the weights. Can be "pls" or "random" or a callable
    tol : float, default=1e-3
        Tolerance for convergence
    alpha : float or list of floats, default=None
        Regularisation parameter for the L2 penalty. If None, defaults to 1.0
    l1_ratio : float or list of floats, default=None
        Regularisation parameter for the L1 penalty. If None, defaults to 0.0
    stochastic : bool, default=False
        Whether to use stochastic gradient descent
    positive : bool or list of bools, default=None
        Whether to use non-negative constraints

    Examples
    --------
    >>> from cca_zoo.models import ElasticCCA
    >>> import numpy as np
    >>> rng=np.random.RandomState(0)
    >>> X1 = rng.random((10,5))
    >>> X2 = rng.random((10,5))
    >>> model = ElasticCCA(c=[1e-1,1e-1],l1_ratio=[0.5,0.5], random_state=0)
    >>> model.fit((X1,X2)).score((X1,X2))
    array([0.9316638])
    """

    def __init__(
        self,
        latent_dims: int = 1,
        copy_data=True,
        random_state=None,
        epochs: int = 100,
        deflation="cca",
        initialization: Union[str, callable] = "pls",
        tol: float = 1e-3,
        alpha: Union[Iterable[float], float] = None,
        l1_ratio: Union[Iterable[float], float] = None,
        stochastic=False,
        positive: Union[Iterable[bool], bool] = None,
        convergence_checking=None,
        track=None,
        verbose=False,
    ):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.stochastic = stochastic
        self.positive = positive
        if self.positive is not None and stochastic:
            self.stochastic = False
            warnings.warn(
                "Non negative constraints cannot be used with _stochastic regressors. Switching to _stochastic=False"
            )
        super().__init__(
            latent_dims=latent_dims,
            copy_data=copy_data,
            epochs=epochs,
            deflation=deflation,
            initialization=initialization,
            tol=tol,
            random_state=random_state,
            convergence_checking=convergence_checking,
            patience=0,
            track=track,
            verbose=verbose,
        )

    def _check_params(self):
        self.alpha = _process_parameter("alpha", self.alpha, 0, self.n_views_)
        self.l1_ratio = _process_parameter("l1_ratio", self.l1_ratio, 0, self.n_views_)
        self.positive = _process_parameter(
            "positive", self.positive, False, self.n_views_
        )

    def _get_module(self, weights=None, k=None):
        return ElasticLoop(
            weights=weights,
            k=k,
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            stochastic=self.stochastic,
            positive=self.positive,
            tol=self.tol,
            random_state=self.random_state,
            tracking=self.track,
            convergence_checking=self.convergence_checking,
        )

    def _more_tags(self):
        return {"multiview": True}


class ElasticLoop(BaseLoop):
    def __init__(
        self,
        weights,
        k=None,
        alpha=None,
        l1_ratio=None,
        stochastic=False,
        positive=None,
        tol=1e-3,
        random_state=None,
        tracking=False,
        convergence_checking=False,
    ):
        super().__init__(
            weights=weights,
            k=k,
            automatic_optimization=False,
            tracking=tracking,
            convergence_checking=convergence_checking,
        )
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.n_views = len(self.weights)
        self.n_samples_ = self.weights[0].shape[0]
        self.regressors = initialize_regressors(
            alpha,
            l1_ratio,
            positive,
            stochastic,
            tol,
            random_state,
        )
        for view_index, weight in enumerate(weights):
            self.regressors[view_index].coef_ = weight[:, 0]
            self.regressors[view_index].intercept_ = 0

    def forward(self, views: list) -> list:
        scores = []
        for view_index, view in enumerate(views):
            scores.append(self.regressors[view_index].predict(view))
        return scores

    def training_step(self, batch, batch_idx):
        scores = np.stack(self(batch["views"]))
        old_weights = self.weights.copy()
        # Update each view using loop update function
        for view_index, view in enumerate(batch["views"]):
            # create a mask that is True for elements not equal to k along dim k
            mask = np.arange(scores.shape[0]) != view_index
            # apply the mask to scores and sum along dim k
            target = np.sum(scores[mask], axis=0)
            target /= np.linalg.norm(target) / np.sqrt(self.n_samples_)
            # Solve the elastic regression
            self.regressors[view_index] = self.regressors[view_index].fit(
                batch["views"][view_index], target
            )
            self.weights[view_index] = self.regressors[view_index].coef_
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

    def on_fit_end(self) -> None:
        self.weights = [regressor.coef_ for regressor in self.regressors]

    def objective(self, views):
        scores = np.stack(self(views))
        objective = 0
        for view_index, view in enumerate(views):
            # create a mask that is True for elements not equal to k along dim k
            mask = np.arange(scores.shape[0]) != view_index
            # apply the mask to scores and sum along dim k
            target = np.sum(scores[mask], axis=0)
            target /= np.linalg.norm(target) / np.sqrt(self.n_samples_)
            objective += elastic_objective(
                view,
                self.regressors[view_index].coef_,
                target,
                self.alpha[view_index],
                self.l1_ratio[view_index],
            )
        return objective


class SCCA_IPLS(ElasticCCA):
    r"""
    A class used to fit a sparse CCA model by iterative rescaled lasso regression. Implemented by ElasticCCA with l1 ratio=1 and SUMCOR form.

    This model finds the linear projections of multiple views that maximize their pairwise correlations while enforcing sparsity constraints on the projection vectors.

    The objective function of sparse CCA is:

    .. math::

        w_{opt}=\underset{w}{\mathrm{argmin}}\{\sum_i\sum_{j\neq i} \|X_iw_i-X_jw_j\|^2 + \sum_i \alpha_i\|w_i\|_1 \}\\

        \text{subject to:}

        w_i^TX_i^TX_iw_i=n

    where :math:`\alpha_i` are the sparsity parameters for each view.

    This model uses the SUMCOR form of CCA, which means that the projection vectors are normalized by their auto-covariance matrices.



    """

    def _get_module(self, weights=None, k=None):
        self.l1_ratio = [1] * self.n_views_
        return IPLSLoop(
            weights=weights,
            k=k,
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            stochastic=self.stochastic,
            positive=self.positive,
            tol=self.tol,
            random_state=self.random_state,
        )


class IPLSLoop(ElasticLoop):
    def training_step(self, batch, batch_idx):
        scores = np.stack(self(batch["views"]))
        old_weights = self.weights.copy()
        # Update each view using loop update function
        for view_index, view in enumerate(batch["views"]):
            # create a mask that is True for elements not equal to k along dim k
            mask = np.arange(scores.shape[0]) != view_index
            # apply the mask to scores and sum along dim k
            target = np.sum(scores[mask], axis=0)
            # Solve the elastic regression
            self.regressors[view_index] = self.regressors[view_index].fit(
                batch["views"][view_index], target
            )
            self.regressors[view_index].coef_ /= np.linalg.norm(
                view @ self.regressors[view_index].coef_
            ) / np.sqrt(self.n_samples_)
            self.weights[view_index] = self.regressors[view_index].coef_
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
                return {"loss": objective, "weights_change": weights_change}

    def objective(self, views):
        scores = np.stack(self(views))

        objective = 0
        for view_index, view in enumerate(views):
            # create a mask that is True for elements not equal to k along dim k
            mask = np.arange(scores.shape[0]) != view_index
            # apply the mask to scores and sum along dim k
            target = np.sum(scores[mask], axis=0)
            target /= np.linalg.norm(target) / np.sqrt(self.n_samples_)
            objective += elastic_objective(
                view,
                self.regressors[view_index].coef_,
                target,
                self.alpha[view_index],
                self.l1_ratio[view_index],
            )
        return {"loss": torch.tensor(objective)}


def elastic_objective(x, w, y, alpha, l1_ratio):
    n = len(y)
    z = x @ w
    objective = np.linalg.norm(z - y) ** 2 / (2 * n)
    l1_pen = alpha * l1_ratio * np.linalg.norm(w, ord=1)
    l2_pen = alpha * (1 - l1_ratio) * np.linalg.norm(w, ord=2)
    return objective + l1_pen + l2_pen


def initialize_regressors(alpha, l1_ratio, positive, stochastic, tol, random_state):
    regressors = []
    for alpha, l1_ratio, positive in zip(alpha, l1_ratio, positive):
        if stochastic:
            regressors.append(
                SGDRegressor(
                    penalty="elasticnet",
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    fit_intercept=False,
                    tol=tol,
                    warm_start=True,
                    random_state=random_state,
                )
            )
        elif l1_ratio == 0:
            regressors.append(
                Ridge(
                    alpha=alpha,
                    fit_intercept=False,
                    positive=positive,
                    random_state=random_state,
                    tol=tol,
                )
            )
        elif l1_ratio == 1:
            regressors.append(
                Lasso(
                    alpha=alpha,
                    fit_intercept=False,
                    warm_start=True,
                    positive=positive,
                    random_state=random_state,
                    tol=tol,
                    selection="random",
                )
            )
        else:
            regressors.append(
                ElasticNet(
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    fit_intercept=False,
                    warm_start=True,
                    positive=positive,
                    random_state=random_state,
                    tol=tol,
                    selection="random",
                )
            )
    return regressors
