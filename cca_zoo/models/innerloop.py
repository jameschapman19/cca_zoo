import warnings
from abc import abstractmethod
from itertools import combinations

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
    ElasticNet,
    SGDRegressor,
)
from sklearn.utils._testing import ignore_warnings
from sklearn.utils.validation import check_random_state

from ..utils.check_values import (
    _check_converged_weights,
    _check_Parikh2014,
    _process_parameter,
)


class _InnerLoop:
    def __init__(
        self,
        max_iter: int = 100,
        tol: float = 1e-5,
        initialization: str = "unregularized",
        random_state=None,
    ):
        """
        :param max_iter: maximum number of iterations to perform if tol is not reached
        :param tol: tolerance value used for stopping criteria
        :param initialization: initialise the optimisation with either the 'unregularized' (CCA/PLS) solution, or a 'random' initialisation
        """
        self.initialization = initialization
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = check_random_state(random_state)

    def _check_params(self):
        """
        Put any parameter checks using exceptions inside this function.
        """
        pass

    def _initialize(self):
        if self.initialization == "random":
            self.scores = np.array(
                [self.random_state.randn(view.shape[0], 1) for view in self.views]
            )
        elif self.initialization == "uniform":
            self.scores = np.array([np.ones((view.shape[0], 1)) for view in self.views])
        elif self.initialization == "unregularized":
            self.scores = (
                PLSInnerLoop(
                    initialization="random",
                    random_state=self.random_state,
                    tol=self.tol,
                )
                ._fit(*self.views)
                .scores
            )
        else:
            raise ValueError("initialize must be random, uniform or unregularized")
        self.scores = (
            self.scores
            * np.sqrt(self.n - 1)
            / np.linalg.norm(self.scores, axis=1)[:, np.newaxis]
        )
        self.weights = [
            self.random_state.randn(view.shape[1], 1) for view in self.views
        ]

    def _fit(self, *views: np.ndarray):
        self.views = views
        self.n = views[0].shape[0]

        # Check that the parameters that have been passed are valid for these views given #views and #features
        self._check_params()
        self._initialize()

        self.track = {}
        self.track["converged"] = False
        # Iterate until convergence
        self.track["objective"] = []
        for _ in range(self.max_iter):
            self._inner_iteration()
            if np.isnan(self.scores).sum() > 0:
                warnings.warn(
                    f"Some scores are nan. Usually regularisation is too high."
                )
                break
            self.track["objective"].append(self._objective())
            if _ > 1 and self._early_stop():
                self.track["converged"] = True
                break
            self.old_scores = self.scores.copy()
        return self

    def _early_stop(self) -> bool:
        return False

    @abstractmethod
    def _inner_iteration(self):
        pass

    def _objective(self) -> int:
        """
        Function used to calculate the objective function for the given. If we do not override then returns the covariance
         between projections

        :return:
        """
        # default objective is correlation
        obj = 0
        for (score_i, score_j) in combinations(self.scores, 2):
            obj += score_i.T @ score_j
        return obj.item()


class PLSInnerLoop(_InnerLoop):
    def __init__(
        self,
        max_iter: int = 100,
        tol=1e-5,
        initialization: str = "unregularized",
        random_state=None,
    ):
        super().__init__(
            max_iter=max_iter,
            tol=tol,
            initialization=initialization,
            random_state=random_state,
        )

    def _check_params(self):
        self.l1_ratio = [0] * len(self.views)
        self.c = [0] * len(self.views)

    def _inner_iteration(self):
        # Update each view using loop update function
        for i, view in enumerate(self.views):
            # if no nans
            if np.isnan(self.scores).sum() == 0:
                self._update_view(i)

    @abstractmethod
    def _update_view(self, view_index: int):
        """
        Function used to update the parameters in each view within the loop. By changing this function, we can change
         the optimisation. This method NEEDS to update self.scores[view_index]

        :param view_index: index of view being updated
        :return: self with updated weights
        """
        # mask off the current view and sum the rest
        targets = np.ma.array(self.scores, mask=False)
        targets.mask[view_index] = True
        self.weights[view_index] = (
            self.views[view_index].T @ targets.sum(axis=0).filled()
        )
        self.weights[view_index] /= np.linalg.norm(self.weights[view_index])
        self.scores[view_index] = self.views[view_index] @ self.weights[view_index]

    def _early_stop(self) -> bool:
        # Some kind of early stopping
        if all(
            _cosine_similarity(self.scores[n], self.old_scores[n]) > (1 - self.tol)
            for n, view in enumerate(self.scores)
        ):
            return True
        else:
            return False


class PMDInnerLoop(PLSInnerLoop):
    def __init__(
        self,
        max_iter: int = 100,
        tol=1e-5,
        initialization: str = "unregularized",
        c=None,
        positive=None,
        random_state=None,
    ):
        super().__init__(
            max_iter=max_iter,
            tol=tol,
            initialization=initialization,
            random_state=random_state,
        )
        self.c = c
        self.positive = positive

    def _check_params(self):
        if self.c is None:
            warnings.warn(
                "c parameter not set. Setting to c=1 i.e. maximum regularisation of l1 norm"
            )
        self.c = _process_parameter("c", self.c, 1, len(self.views))
        if any(c < 1 for c in self.c):
            raise ValueError(
                "All regulariation parameters should be at least " f"1. c=[{self.c}]"
            )
        shape_sqrts = [np.sqrt(view.shape[1]) for view in self.views]
        if any(c > shape_sqrt for c, shape_sqrt in zip(self.c, shape_sqrts)):
            raise ValueError(
                "All regulariation parameters should be less than"
                " the square root of number of the respective"
                f" view. c=[{self.c}], limit of each view: "
                f"{shape_sqrts}"
            )
        self.positive = _process_parameter(
            "positive", self.positive, False, len(self.views)
        )

    def _update_view(self, view_index: int):
        """
        :param view_index: index of view being updated
        :return: updated weights
        """
        # mask off the current view and sum the rest
        targets = np.ma.array(self.scores, mask=False)
        targets.mask[view_index] = True
        self.weights[view_index] = (
            self.views[view_index].T @ targets.sum(axis=0).filled()
        )
        self.weights[view_index] = _delta_search(
            self.weights[view_index],
            self.c[view_index],
            positive=self.positive[view_index],
        )
        _check_converged_weights(self.weights[view_index], view_index)
        self.scores[view_index] = self.views[view_index] @ self.weights[view_index]


class ParkhomenkoInnerLoop(PLSInnerLoop):
    def __init__(
        self,
        max_iter: int = 100,
        tol=1e-5,
        initialization: str = "unregularized",
        c=None,
        random_state=None,
    ):
        super().__init__(
            max_iter=max_iter,
            tol=tol,
            initialization=initialization,
            random_state=random_state,
        )
        self.c = c

    def _check_params(self):
        self.c = _process_parameter("c", self.c, 0.0001, len(self.views))
        if any(c <= 0 for c in self.c):
            raise ("All regularisation parameters should be above 0. " f"c=[{self.c}]")

    def _update_view(self, view_index: int):
        """
        :param view_index: index of view being updated
        :return: updated weights
        """
        # mask off the current view and sum the rest
        targets = np.ma.array(self.scores, mask=False)
        targets.mask[view_index] = True
        w = self.views[view_index].T @ targets.sum(axis=0).filled()
        w /= np.linalg.norm(w)
        _check_converged_weights(w, view_index)
        w = _soft_threshold(w, self.c[view_index] / 2)
        self.weights[view_index] = w / np.linalg.norm(w)
        _check_converged_weights(w, view_index)
        self.scores[view_index] = self.views[view_index] @ self.weights[view_index]


class ElasticInnerLoop(PLSInnerLoop):
    def __init__(
        self,
        max_iter: int = 100,
        tol=1e-5,
        initialization: str = "unregularized",
        c=None,
        l1_ratio=None,
        maxvar=True,
        stochastic=True,
        positive=None,
        random_state=None,
    ):
        super().__init__(
            max_iter=max_iter,
            tol=tol,
            initialization=initialization,
            random_state=random_state,
        )
        self.stochastic = stochastic
        self.c = c
        self.l1_ratio = l1_ratio
        self.positive = positive
        self.maxvar = maxvar

    def _check_params(self):
        self.c = _process_parameter("c", self.c, 0, len(self.views))
        self.l1_ratio = _process_parameter(
            "l1_ratio", self.l1_ratio, 0, len(self.views)
        )
        self.positive = _process_parameter(
            "positive", self.positive, False, len(self.views)
        )
        self.regressors = []
        for alpha, l1_ratio, positive in zip(self.c, self.l1_ratio, self.positive):
            if self.stochastic:
                self.regressors.append(
                    SGDRegressor(
                        penalty="elasticnet",
                        alpha=alpha / len(self.views),
                        l1_ratio=l1_ratio,
                        fit_intercept=False,
                        tol=self.tol,
                        warm_start=True,
                        random_state=self.random_state,
                    )
                )
            else:
                self.regressors.append(
                    ElasticNet(
                        alpha=alpha / len(self.views),
                        l1_ratio=l1_ratio,
                        fit_intercept=False,
                        warm_start=True,
                        positive=positive,
                        random_state=self.random_state,
                    )
                )

    def _update_view(self, view_index: int):
        """
        :param view_index: index of view being updated
        :return: updated weights
        """
        if self.maxvar:
            # For MAXVAR we rescale the targets
            target = self.scores.mean(axis=0)
            target /= np.linalg.norm(target) / np.sqrt(self.n)
        else:
            target = self.scores[view_index - 1]
        # Solve the elastic regression
        self.weights[view_index] = self._elastic_solver(
            self.views[view_index], target, view_index
        )
        # For SUMCOR we rescale the projections
        if not self.maxvar:
            _check_converged_weights(self.weights[view_index], view_index)
            self.weights[view_index] = self.weights[view_index] / (
                np.linalg.norm(self.views[view_index] @ self.weights[view_index])
                / np.sqrt(self.n)
            )
        self.scores[view_index] = self.views[view_index] @ self.weights[view_index]

    @ignore_warnings(category=ConvergenceWarning)
    def _elastic_solver(self, X, y, view_index):
        return np.expand_dims(self.regressors[view_index].fit(X, y.ravel()).coef_, 1)

    def _objective(self):
        views = len(self.views)
        c = np.array(self.c)
        ratio = np.array(self.l1_ratio)
        l1 = c * ratio
        l2 = c * (1 - ratio)
        total_objective = 0
        target = self.scores.mean(axis=0)
        for i in range(views):
            if self.maxvar:
                target /= np.linalg.norm(target) / np.sqrt(self.n)
            objective = np.linalg.norm(
                self.views[i] @ self.weights[i] - target
            ) ** 2 / (2 * self.n)
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


class ADMMInnerLoop(ElasticInnerLoop):
    def __init__(
        self,
        max_iter: int = 100,
        tol=1e-5,
        initialization: str = "unregularized",
        mu=None,
        lam=None,
        c=None,
        eta=None,
        random_state=None,
    ):
        super().__init__(
            max_iter=max_iter,
            tol=tol,
            initialization=initialization,
            random_state=random_state,
        )
        self.c = c
        self.lam = lam
        self.mu = mu
        self.eta = eta

    def _check_params(self):
        self.c = _process_parameter("c", self.c, 0, len(self.views))
        self.lam = _process_parameter("lam", self.lam, 1, len(self.views))
        if self.mu is None:
            self.mu = [
                lam / np.linalg.norm(view) ** 2
                for lam, view in zip(self.lam, self.views)
            ]
        else:
            self.mu = _process_parameter("mu", self.mu, 0, len(self.views))
        self.eta = _process_parameter("eta", self.eta, 0, len(self.views))

        if any(mu <= 0 for mu in self.mu):
            raise ValueError("At least one mu is less than zero.")

        _check_Parikh2014(self.mu, self.lam, self.views)

        self.eta = [
            np.ones((view.shape[0], 1)) * eta for view, eta in zip(self.views, self.eta)
        ]
        self.z = [np.zeros((view.shape[0], 1)) for view in self.views]
        self.l1_ratio = [1] * len(self.views)

    def _update_view(self, view_index: int):
        targets = np.ma.array(self.scores, mask=False)
        targets.mask[view_index] = True
        # Suo uses parameter tau whereas we use parameter c to penalize the 1-norm of the weights.
        # Suo uses c to refer to the gradient where we now use gradient
        gradient = self.views[view_index].T @ targets.sum(axis=0).filled()
        # reset eta each loop?
        # self.eta[view_index][:] = 0
        mu = self.mu[view_index]
        lam = self.lam[view_index]
        N = self.views[view_index].shape[0]
        unnorm_z = []
        norm_eta = []
        norm_weights = []
        norm_proj = []
        for _ in range(self.max_iter):
            # We multiply 'c' by N in order to make regularisation match across the different sparse cca methods
            self.weights[view_index] = self._prox_mu_f(
                self.weights[view_index]
                - mu
                / lam
                * self.views[view_index].T
                @ (
                    self.views[view_index] @ self.weights[view_index]
                    - self.z[view_index]
                    + self.eta[view_index]
                ),
                mu,
                gradient,
                N * self.c[view_index],
            )
            unnorm_z.append(
                np.linalg.norm(
                    self.views[view_index] @ self.weights[view_index]
                    + self.eta[view_index]
                )
            )
            self.z[view_index] = self._prox_lam_g(
                self.views[view_index] @ self.weights[view_index] + self.eta[view_index]
            )
            self.eta[view_index] = (
                self.eta[view_index]
                + self.views[view_index] @ self.weights[view_index]
                - self.z[view_index]
            )
            norm_eta.append(np.linalg.norm(self.eta[view_index]))
            norm_proj.append(
                np.linalg.norm(self.views[view_index] @ self.weights[view_index])
            )
            norm_weights.append(np.linalg.norm(self.weights[view_index], 1))
        _check_converged_weights(self.weights[view_index], view_index)
        self.scores[view_index] = self.views[view_index] @ self.weights[view_index]

    def _prox_mu_f(self, x, mu, c, tau):
        u_update = x.copy()
        mask_1 = x + (mu * c) > mu * tau
        # if mask_1.sum()>0:
        u_update[mask_1] = x[mask_1] + mu * (c[mask_1] - tau)
        mask_2 = x + (mu * c) < -mu * tau
        # if mask_2.sum() > 0:
        u_update[mask_2] = x[mask_2] + mu * (c[mask_2] + tau)
        mask_3 = ~(mask_1 | mask_2)
        u_update[mask_3] = 0
        return u_update

    def _prox_lam_g(self, x):
        norm = np.linalg.norm(x)
        if norm < 1:
            return x
        else:
            return x / max(1, norm)


class SpanCCAInnerLoop(_InnerLoop):
    def __init__(
        self,
        max_iter: int = 100,
        tol=1e-5,
        initialization: str = "unregularized",
        c=None,
        regularisation="l0",
        rank=1,
        random_state=None,
        positive=False,
    ):
        super().__init__(
            max_iter=max_iter,
            tol=tol,
            initialization=initialization,
            random_state=random_state,
        )
        self.c = c
        self.regularisation = regularisation
        self.rank = rank
        self.positive = positive

    def _check_params(self):
        """check number of views=2"""
        if len(self.views) != 2:
            raise ValueError(f"SpanCCA requires only 2 views")
        cov = self.views[0].T @ self.views[1] / self.n
        # Perform SVD on im and obtain individual matrices
        P, D, Q = np.linalg.svd(cov, full_matrices=True)
        self.P = P[:, : self.rank]
        self.D = D[: self.rank]
        self.Q = Q[: self.rank, :].T
        self.max_obj = 0
        if self.regularisation == "l0":
            self.update = _support_soft_thresh
            self.c = _process_parameter("c", self.c, 0, len(self.views))
        elif self.regularisation == "l1":
            self.update = _delta_search
            self.c = _process_parameter("c", self.c, 0, len(self.views))
        self.positive = _process_parameter(
            "positive", self.positive, False, len(self.views)
        )

    def _inner_iteration(self):
        c = self.random_state.randn(self.rank, 1)
        c /= np.linalg.norm(c)
        a = self.P @ np.diag(self.D) @ c
        u = self.update(a, self.c[0])
        u /= np.linalg.norm(u)
        b = self.Q @ np.diag(self.D) @ self.P.T @ u
        v = self.update(b, self.c[1])
        v /= np.linalg.norm(v)
        if b.T @ v > self.max_obj:
            self.max_obj = b.T @ v
            self.scores[0] = self.views[0] @ u
            self.scores[1] = self.views[1] @ v
            self.weights[0] = u
            self.weights[1] = v


class SWCCAInnerLoop(PLSInnerLoop):
    def __init__(
        self,
        max_iter: int = 100,
        tol=1e-20,
        initialization: str = "unregularized",
        regularisation="l0",
        c=None,
        sample_support: int = None,
        random_state=None,
        positive=False,
    ):
        super().__init__(
            max_iter=max_iter,
            tol=tol,
            initialization=initialization,
            random_state=random_state,
        )
        self.c = c
        self.sample_support = sample_support
        if regularisation == "l0":
            self.update = _support_soft_thresh
        elif regularisation == "l1":
            self.update = _delta_search
        self.positive = positive

    def _check_params(self):
        if self.sample_support is None:
            self.sample_support = self.views[0].shape[0]
        self.sample_weights = np.ones((self.views[0].shape[0], 1))
        self.sample_weights /= np.linalg.norm(self.sample_weights)
        self.c = _process_parameter("c", self.c, 2, len(self.views))
        self.positive = _process_parameter(
            "positive", self.positive, False, len(self.views)
        )

    def _update_view(self, view_index: int):
        """
        :param view_index: index of view being updated
        :return: updated weights
        """
        targets = np.ma.array(self.scores, mask=False)
        targets.mask[view_index] = True
        self.weights[view_index] = (
            self.views[view_index] * self.sample_weights
        ).T @ targets.sum(axis=0).filled()
        self.weights[view_index] = self.update(
            self.weights[view_index],
            self.c[view_index],
            positive=self.positive[view_index],
        )
        self.weights[view_index] /= np.linalg.norm(self.weights[view_index])
        if view_index == len(self.views) - 1:
            self._update_sample_weights()
        self.scores[view_index] = self.views[view_index] @ self.weights[view_index]

    def _update_sample_weights(self):
        w = self.scores.prod(axis=0)
        self.sample_weights = _support_soft_thresh(w, self.sample_support)
        self.sample_weights /= np.linalg.norm(self.sample_weights)
        self.track["sample_weights"] = self.sample_weights

    def _early_stop(self) -> bool:
        return False

    def _objective(self) -> int:
        """
        Function used to calculate the objective function for the given. If we do not override then returns the covariance
         between projections

        :return:
        """
        # default objective is correlation
        obj = 0
        for (score_i, score_j) in combinations(self.scores, 2):
            obj += (score_i * self.sample_weights).T @ score_j
        return obj


def _bin_search(current, previous, current_val, previous_val, min_, max_):
    """Binary search helper function:
    current:current parameter value
    previous:previous parameter value
    current_val:current function value
    previous_val: previous function values
    min_:minimum parameter value resulting in function value less than zero
    max_:maximum parameter value resulting in function value greater than zero
    Problem needs to be set up so that greater parameter, greater target
    """
    if previous_val is None:
        previous_val = current_val
    if current_val <= 0:
        if previous_val <= 0:
            new = (current + max_) / 2
        if previous_val > 0:
            new = (current + previous) / 2
        if current > min_:
            min_ = current
    if current_val > 0:
        if previous_val > 0:
            new = (current + min_) / 2
        if previous_val <= 0:
            new = (current + previous) / 2
        if current < max_:
            max_ = current
    return new, current, min_, max_


def _delta_search(w, c, positive=False, init=0):
    """
    Searches for threshold delta such that the 1-norm of weights w is less than or equal to c
    :param w: weights found by one power method iteration
    :param c: 1-norm threshold
    :return: updated weights
    """
    # First normalise the weights unit length
    w = w / np.linalg.norm(w, 2)
    converged = False
    min_ = 0
    max_ = 10
    current = init
    previous = current
    previous_val = None
    i = 0
    while not converged:
        i += 1
        coef = _soft_threshold(w, current, positive=positive)
        if np.linalg.norm(coef) > 0:
            coef /= np.linalg.norm(coef)
        current_val = c - np.linalg.norm(coef, 1)
        current, previous, min_, max_ = _bin_search(
            current, previous, current_val, previous_val, min_, max_
        )
        previous_val = current_val
        if np.abs(current_val) < 1e-5 or np.abs(max_ - min_) < 1e-30 or i == 50:
            converged = True
    return coef


def _soft_threshold(x, threshold, positive=False):
    """
    if absolute value of x less than threshold replace with zero
    :param x: input
    :return: x soft-thresholded by threshold
    """
    if positive:
        u = np.clip(x, 0, None)
    else:
        u = np.abs(x)
    u = u - threshold
    u[u < 0] = 0
    return u * np.sign(x)


def _support_soft_thresh(x, support, positive=False):
    if x.shape[0] <= support or np.linalg.norm(x) == 0:
        return x
    if positive:
        u = np.clip(x, 0, None)
    else:
        u = np.abs(x)
    idx = np.argpartition(x.ravel(), x.shape[0] - support)
    u[idx[:-support]] = 0
    return u * np.sign(x)


def _cosine_similarity(a, b):
    """
    Calculates the cosine similarity between vectors
    :param a: 1d numpy array
    :param b: 1d numpy array
    :return: cosine similarity
    """
    # https: // www.statology.org / cosine - similarity - python /
    return a.T @ b / (np.linalg.norm(a) * np.linalg.norm(b))
