import warnings
from typing import Union, Iterable

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDRegressor, Ridge, ElasticNet, Lasso
from sklearn.utils._testing import ignore_warnings

from cca_zoo.models._iterative._base import _BaseIterative
from cca_zoo.utils import _process_parameter, _check_converged_weights


class ElasticCCA(_BaseIterative):
    r"""
    Fits an elastic CCA by iterating elastic net regressions to two or more views of data.

    By default, ElasticCCA uses CCA with an auxiliary variable target i.e. MAXVAR configuration

    .. math::

        w_{opt}, t_{opt}=\underset{w,t}{\mathrm{argmax}}\{\sum_i \|X_iw_i-t\|^2 + c\|w_i\|^2_2 + \text{l1_ratio}\|w_i\|_1\}\\

        \text{subject to:}

        t^Tt=n

    But we can force it to attempt to use the SUMCOR form which will approximate a solution to the problem:

    .. math::

        w_{opt}=\underset{w}{\mathrm{argmax}}\{\sum_i\sum_{j\neq i} \|X_iw_i-X_jw_j\|^2 + c\|w_i\|^2_2 + \text{l1_ratio}\|w_i\|_1\}\\

        \text{subject to:}

        w_i^TX_i^TX_iw_i=n

    Parameters
    ----------
    latent_dims : int, default=1
        Number of latent dimensions to use
    scale : bool, default=True
        Whether to scale the data to unit variance
    centre : bool, default=True
        Whether to centre the data to zero mean
    copy_data : bool, default=True
        Whether to copy the data or overwrite it
    random_state : int, default=None
        Random seed for initialization
    deflation : str, default="cca"
        Whether to use CCA or PLS deflation
    max_iter : int, default=100
        Maximum number of iterations to run
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
    verbose : int, default=0
        Verbosity level


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
        scale: bool = True,
        centre=True,
        copy_data=True,
        random_state=None,
        deflation="cca",
        max_iter: int = 100,
        initialization: Union[str, callable] = "pls",
        tol: float = 1e-3,
        alpha: Union[Iterable[float], float] = None,
        l1_ratio: Union[Iterable[float], float] = None,
        stochastic=False,
        positive: Union[Iterable[bool], bool] = None,
        verbose=0,
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
            scale=scale,
            centre=centre,
            copy_data=copy_data,
            deflation=deflation,
            max_iter=max_iter,
            initialization=initialization,
            tol=tol,
            random_state=random_state,
            verbose=verbose,
        )

    def _check_params(self):
        self.alpha = _process_parameter("alpha", self.alpha, 0, self.n_views)
        self.l1_ratio = _process_parameter("l1_ratio", self.l1_ratio, 0, self.n_views)
        self.positive = _process_parameter(
            "positive", self.positive, False, self.n_views
        )

    def _initialize(self, views):
        self.regressors = initialize_regressors(
            self.alpha,
            self.l1_ratio,
            self.positive,
            self.stochastic,
            self.tol,
            self.random_state,
        )

    def _update(self, views, scores, weights):
        for view_index, view in enumerate(views):
            # For MAXVAR we rescale the targets
            target = scores.mean(axis=0)
            target /= np.linalg.norm(target) / np.sqrt(self.n)
            # Solve the elastic regression
            weights[view_index] = self._elastic_solver(
                views[view_index], target, view_index
            )
            scores[view_index] = views[view_index] @ weights[view_index]
        return scores, weights

    @ignore_warnings(category=ConvergenceWarning)
    def _elastic_solver(self, X, y, view_index):
        return self.regressors[view_index].fit(X, y.ravel()).coef_

    def _objective(self, views, scores, weights) -> int:
        alpha = np.array(self.alpha)
        l1_ratio = np.array(self.l1_ratio)
        total_objective = 0
        for i, (view, weight) in enumerate(zip(views, weights)):
            target = scores.mean(axis=0)
            target /= np.linalg.norm(target) / np.sqrt(self.n)
            total_objective += elastic_objective(
                view, weight, target, alpha[i], l1_ratio[i]
            )
        return total_objective

    def _get_target(self, scores):
        target = scores.mean(axis=0)
        target /= np.linalg.norm(target) / np.sqrt(scores.shape[0])
        return target

    def _more_tags(self):
        return {"multiview": True}


class SCCA_IPLS(ElasticCCA):
    r"""
    Fits a sparse CCA model by _iterative rescaled lasso regression. Implemented by ElasticCCA with l1 ratio=1

    The optimisation is given by:

    :Maths:

    .. math::

        w_{opt}=\underset{w}{\mathrm{argmax}}\{\sum_i\sum_{j\neq i} \|X_iw_i-X_jw_j\|^2 + \text{l1_ratio}\|w_i\|_1\}\\

        \text{subject to:}

        w_i^TX_i^TX_iw_i=n

    :Citation:

    Mai, Qing, and Xin Zhang. "An _iterative penalized least squares approach to sparse canonical correlation analysis." Biometrics 75.3 (2019): 734-744.


    :Example:

    >>> from cca_zoo.models import SCCA_IPLS
    >>> import numpy as np
    >>> rng=np.random.RandomState(0)
    >>> X1 = rng.random((10,5))
    >>> X2 = rng.random((10,5))
    >>> model = SCCA_IPLS(c=[0.001,0.001], random_state=0)
    >>> model.fit((X1,X2)).score((X1,X2))
    array([0.99998761])
    """

    def __init__(
        self,
        latent_dims: int = 1,
        scale: bool = True,
        centre=True,
        copy_data=True,
        random_state=None,
        deflation="cca",
        tau: Union[Iterable[float], float] = None,
        max_iter: int = 100,
        initialization: Union[str, callable] = "pls",
        tol: float = 1e-3,
        stochastic=False,
        positive: Union[Iterable[bool], bool] = None,
        verbose=0,
    ):
        self.tau = tau
        super().__init__(
            latent_dims=latent_dims,
            scale=scale,
            centre=centre,
            copy_data=copy_data,
            max_iter=max_iter,
            initialization=initialization,
            tol=tol,
            stochastic=stochastic,
            positive=positive,
            random_state=random_state,
            deflation=deflation,
            verbose=verbose,
            l1_ratio=1,
        )

    def _initialize(self, views):
        self.regressors = initialize_regressors(
            self.tau,
            self.l1_ratio,
            self.positive,
            self.stochastic,
            self.tol,
            self.random_state,
        )
        self.alpha = self.tau

    def _check_params(self):
        self.tau = _process_parameter("tau", self.tau, 0, self.n_views)
        self.l1_ratio = _process_parameter("l1_ratio", self.l1_ratio, 0, self.n_views)
        self.positive = _process_parameter(
            "positive", self.positive, False, self.n_views
        )

    def _update(self, views, scores, weights):
        for view_index, view in enumerate(views):
            target = scores[view_index - 1]
            # Solve the elastic regression
            weights[view_index] = self._elastic_solver(
                views[view_index], target, view_index
            )
            _check_converged_weights(weights[view_index], view_index)
            weights[view_index] = weights[view_index] / (
                np.linalg.norm(views[view_index] @ weights[view_index])
                / np.sqrt(self.n)
            )
            scores[view_index] = views[view_index] @ weights[view_index]
        return scores, weights

    def _objective(self, views, scores, weights) -> int:
        tau = np.array(self.tau)
        total_objective = 0
        for i, (view, weight) in enumerate(zip(views, weights)):
            total_objective += elastic_objective(view, weight, scores[i - 1], tau[i], 1)
        return total_objective


def elastic_objective(x, w, y, alpha, l1_ratio):
    n = len(y)
    z = x @ w
    objective = np.linalg.norm(z - y) ** 2 / (2 * n)
    l1_pen = alpha * l1_ratio * np.linalg.norm(w, ord=1)
    l2_pen = alpha * (1 - l1_ratio) * np.linalg.norm(w, ord=2)
    return objective + l1_pen + l2_pen


def initialize_regressors(c, l1_ratio, positive, stochastic, tol, random_state):
    regressors = []
    for alpha, l1_ratio, positive in zip(c, l1_ratio, positive):
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
