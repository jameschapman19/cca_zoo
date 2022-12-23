import warnings
from typing import Union, Iterable

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDRegressor, Ridge, ElasticNet
from sklearn.utils._testing import ignore_warnings

from cca_zoo.utils import _process_parameter, _check_converged_weights
from ._base import _BaseIterative
from ._pls_als import _PLSInnerLoop


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
    tol : float, default=1e-9
        Tolerance for convergence
    c : float or list of floats, default=None
        Regularisation parameter for the L2 penalty. If None, defaults to 1.0
    l1_ratio : float or list of floats, default=None
        Regularisation parameter for the L1 penalty. If None, defaults to 0.0
    maxvar : bool, default=True
        Whether to use MAXVAR or SUMCOR configuration
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
        tol: float = 1e-9,
        c: Union[Iterable[float], float] = None,
        l1_ratio: Union[Iterable[float], float] = None,
        maxvar: bool = True,
        stochastic=False,
        positive: Union[Iterable[bool], bool] = None,
        verbose=0,
    ):
        self.c = c
        self.l1_ratio = l1_ratio
        self.maxvar = maxvar
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

    def _set_loop_params(self):
        self.loop = _ElasticInnerLoop(
            max_iter=self.max_iter,
            c=self.c,
            l1_ratio=self.l1_ratio,
            maxvar=self.maxvar,
            tol=self.tol,
            stochastic=self.stochastic,
            positive=self.positive,
            random_state=self.random_state,
            verbose=self.verbose,
        )

    def _check_params(self):
        self.c = _process_parameter("c", self.c, 0, self.n_views)
        self.l1_ratio = _process_parameter("l1_ratio", self.l1_ratio, 0, self.n_views)
        self.positive = _process_parameter(
            "positive", self.positive, False, self.n_views
        )


class SCCA_IPLS(ElasticCCA):
    r"""
    Fits a sparse CCA model by _iterative rescaled lasso regression. Implemented by ElasticCCA with l1 ratio=1

    For default maxvar=False, the optimisation is given by:

    :Maths:

    .. math::

        w_{opt}=\underset{w}{\mathrm{argmax}}\{\sum_i\sum_{j\neq i} \|X_iw_i-X_jw_j\|^2 + \text{l1_ratio}\|w_i\|_1\}\\

        \text{subject to:}

        w_i^TX_i^TX_iw_i=n

    :Citation:

    Mai, Qing, and Xin Zhang. "An _iterative penalized least squares approach to sparse canonical correlation analysis." Biometrics 75.3 (2019): 734-744.

    For maxvar=True, the optimisation is given by the ElasticCCA problem with no l2 regularisation:

    :Maths:

    .. math::

        w_{opt}, t_{opt}=\underset{w,t}{\mathrm{argmax}}\{\sum_i \|X_iw_i-t\|^2 + \text{l1_ratio}\|w_i\|_1\}\\

        \text{subject to:}

        t^Tt=n

    :Citation:

    Fu, Xiao, et al. "Scalable and flexible multiview MAX-VAR canonical correlation analysis." IEEE Transactions on Signal Processing 65.16 (2017): 4150-4165.


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
        c: Union[Iterable[float], float] = None,
        max_iter: int = 100,
        maxvar: bool = False,
        initialization: Union[str, callable] = "pls",
        tol: float = 1e-9,
        stochastic=False,
        positive: Union[Iterable[bool], bool] = None,
        verbose=0,
    ):
        super().__init__(
            latent_dims=latent_dims,
            scale=scale,
            centre=centre,
            copy_data=copy_data,
            max_iter=max_iter,
            initialization=initialization,
            tol=tol,
            c=c,
            l1_ratio=1,
            maxvar=maxvar,
            stochastic=stochastic,
            positive=positive,
            random_state=random_state,
            deflation=deflation,
            verbose=verbose,
        )


class _ElasticInnerLoop(_PLSInnerLoop):
    def __init__(
        self,
        max_iter: int = 100,
        tol=1e-9,
        c=None,
        l1_ratio=None,
        maxvar=True,
        stochastic=True,
        positive=None,
        random_state=None,
        verbose=0,
        **kwargs,
    ):
        super().__init__(
            max_iter=max_iter, tol=tol, random_state=random_state, verbose=verbose
        )
        self.stochastic = stochastic
        self.c = c
        self.l1_ratio = l1_ratio
        self.positive = positive
        self.maxvar = maxvar

    def _initialize(self, views):
        self.regressors = []
        for alpha, l1_ratio, positive in zip(self.c, self.l1_ratio, self.positive):
            if self.stochastic:
                self.regressors.append(
                    SGDRegressor(
                        penalty="elasticnet",
                        alpha=alpha,
                        l1_ratio=l1_ratio,
                        fit_intercept=False,
                        tol=self.tol,
                        warm_start=True,
                        random_state=self.random_state,
                    )
                )
            elif alpha == 0:
                self.regressors.append(
                    Ridge(
                        alpha=self.tol,
                        fit_intercept=False,
                        positive=positive,
                    )
                )
            else:
                self.regressors.append(
                    ElasticNet(
                        alpha=alpha,
                        l1_ratio=l1_ratio,
                        fit_intercept=False,
                        warm_start=True,
                        positive=positive,
                        random_state=self.random_state,
                    )
                )

    def _update_view(self, views, view_index: int):
        if self.maxvar:
            # For MAXVAR we rescale the targets
            target = self.scores.mean(axis=0)
            target /= np.linalg.norm(target) / np.sqrt(self.n)
        else:
            target = self.scores[view_index - 1]
        # Solve the elastic regression
        self.weights[view_index] = self._elastic_solver(
            views[view_index], target, view_index
        )
        # For SUMCOR we rescale the projections
        if not self.maxvar:
            _check_converged_weights(self.weights[view_index], view_index)
            self.weights[view_index] = self.weights[view_index] / (
                np.linalg.norm(views[view_index] @ self.weights[view_index])
                / np.sqrt(self.n)
            )
        self.scores[view_index] = views[view_index] @ self.weights[view_index]

    @ignore_warnings(category=ConvergenceWarning)
    def _elastic_solver(self, X, y, view_index):
        return self.regressors[view_index].fit(X, y.ravel()).coef_

    def _objective(self, views):
        c = np.array(self.c)
        ratio = np.array(self.l1_ratio)
        l1 = c * ratio
        l2 = c * (1 - ratio)
        total_objective = 0
        target = self.scores.mean(axis=0)
        for i, _ in enumerate(views):
            if self.maxvar:
                target /= np.linalg.norm(target) / np.sqrt(self.n)
            objective = np.linalg.norm(views[i] @ self.weights[i] - target) ** 2 / (
                2 * self.n
            )
            l1_pen = l1[i] * np.linalg.norm(self.weights[i], ord=1)
            l2_pen = l2[i] * np.linalg.norm(self.weights[i], ord=2)
            total_objective += objective + l1_pen + l2_pen
        return total_objective

    def _early_stop(self) -> bool:
        # Some kind of early stopping
        if np.abs(self.track["objective"][-2] - self.track["objective"][-1]) < self.tol:
            return True
        else:
            return False
