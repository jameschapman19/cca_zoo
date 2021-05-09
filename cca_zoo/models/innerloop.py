from abc import abstractmethod
from itertools import combinations

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge, SGDRegressor
from sklearn.utils._testing import ignore_warnings


class _InnerLoop:
    def __init__(self, max_iter: int = 100, tol: float = 1e-5, generalized: bool = False,
                 initialization: str = 'unregularized'):
        """
        :param max_iter: maximum number of iterations to perform if tol is not reached
        :param tol: tolerance value used for stopping criteria
        :param generalized: use an auxiliary variable to
        :param initialization: initialise the optimisation with either the 'unregularized' (CCA/PLS) solution, or a 'random' initialisation
        """
        self.generalized = generalized
        self.initialization = initialization
        self.max_iter = max_iter
        self.tol = tol

    def check_params(self):
        self.l1_ratio = [0] * len(self.views)
        self.c = [0] * len(self.views)

    def fit(self, *views):
        self.views = views
        if len(self.views) > 2:
            self.generalized = True
        self.track_objective = []
        self.track_correlation = []
        self.check_params()
        self.weights = [np.random.rand(view.shape[1], 1) for view in self.views]
        self.weights = [weights / np.linalg.norm(view @ weights) for (weights, view) in
                        zip(self.weights, self.views)]
        if self.initialization == 'random':
            self.scores = np.array([np.random.rand(view.shape[0]) for view in self.views])
        elif self.initialization == 'unregularized':
            unregularized = PLSInnerLoop(initialization='random').fit(*self.views)
            norms = np.linalg.norm(unregularized.scores, axis=1)
            self.scores = unregularized.scores / norms[:, np.newaxis]

        # Iterate until convergence
        for _ in range(self.max_iter):
            # Update each view using loop update function
            for i, view in enumerate(views):
                self.update_view(i)

            self.track_objective.append(self.objective())
            # Some kind of early stopping
            if _ > 0 and all(_cosine_similarity(self.scores[n], self.old_scores[n]) > (1 - self.tol) for n, view in
                             enumerate(self.scores)):
                break
            self.old_scores = self.scores.copy()
        return self

    @abstractmethod
    def update_view(self, view_index: int):
        """
        Function used to update the parameters in each view within the loop. By changing this function, we can change
         the optimisation. This method NEEDS to update self.scores[view_index]

        :param view_index: index of view being updated
        :return: self with updated weights
        """
        targets = np.ma.array(self.scores, mask=False)
        targets.mask[view_index] = True
        self.weights[view_index] = self.views[view_index].T @ targets.sum(axis=0).filled()
        self.weights[view_index] /= np.linalg.norm(self.weights[view_index])
        self.scores[view_index] = self.views[view_index] @ self.weights[view_index]

    def objective(self):
        """
        Function used to calculate the objective function for the given. If we do not override then returns the covariance
         between projections

        :return:
        """
        obj = 0
        for (score_i, score_j) in combinations(self.scores, 2):
            obj += score_i.T @ score_j
        return obj


class PLSInnerLoop(_InnerLoop):
    def __init__(self, max_iter: int = 100, tol=1e-5, generalized: bool = False,
                 initialization: str = 'unregularized'):
        super().__init__(max_iter=max_iter, tol=tol, generalized=generalized,
                         initialization=initialization)


class PMDInnerLoop(_InnerLoop):
    def __init__(self, max_iter: int = 100, tol=1e-5, generalized: bool = False,
                 initialization: str = 'unregularized', c=None):
        super().__init__(max_iter=max_iter, tol=tol, generalized=generalized,
                         initialization=initialization)
        self.c = c

    def check_params(self):
        if self.c is None:
            self.c = [1] * len(self.views)
        assert (all([c >= 1 for c in self.c]))
        assert (all([c <= np.sqrt(view.shape[1]) for c, view in zip(self.c, self.views)]))

    def update_view(self, view_index: int):
        """
        :param view_index: index of view being updated
        :return: updated weights
        """
        targets = np.ma.array(self.scores, mask=False)
        targets.mask[view_index] = True
        self.weights[view_index] = self.views[view_index].T @ targets.sum(axis=0).filled()
        self.weights[view_index], w_success = self.delta_search(self.weights[view_index], self.c[view_index])
        assert (np.linalg.norm(
            self.weights[view_index]) > 0), 'all weights zero. try less regularisation or another initialisation'
        self.weights[view_index] = self.weights[view_index] / np.linalg.norm(self.weights[view_index])
        self.scores[view_index] = self.views[view_index] @ self.weights[view_index]

    def delta_search(self, w, c, init=0):
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
            coef = _soft_threshold(w, current)
            if np.linalg.norm(coef) > 0:
                coef /= np.linalg.norm(coef)
            current_val = c - np.linalg.norm(coef, 1)
            current, previous, min_, max_ = _bin_search(current, previous, current_val, previous_val, min_, max_)
            previous_val = current_val
            if np.abs(current_val) < 1e-5 or np.abs(max_ - min_) < 1e-30 or i == 50:
                converged = True
        return coef, current


class ParkhomenkoInnerLoop(_InnerLoop):
    def __init__(self, max_iter: int = 100, tol=1e-5, generalized: bool = False,
                 initialization: str = 'unregularized', c=None):
        super().__init__(max_iter=max_iter, tol=tol, generalized=generalized,
                         initialization=initialization)
        self.c = c

    def check_params(self):
        if self.c is None:
            self.c = [0.0001] * len(self.views)
        assert (all([c > 0 for c in self.c]))

    def update_view(self, view_index: int):
        """
        :param view_index: index of view being updated
        :return: updated weights
        """
        targets = np.ma.array(self.scores, mask=False)
        targets.mask[view_index] = True
        w = self.views[view_index].T @ targets.sum(axis=0).filled()
        assert (np.linalg.norm(w) > 0), 'all weights zero. try less regularisation or another initialisation'
        w /= np.linalg.norm(w)
        w = _soft_threshold(w, self.c[view_index] / 2)
        assert (np.linalg.norm(w) > 0), 'all weights zero. try less regularisation or another initialisation'
        self.weights[view_index] = w / np.linalg.norm(w)
        self.scores[view_index] = self.views[view_index] @ self.weights[view_index]


class ElasticInnerLoop(_InnerLoop):
    def __init__(self, max_iter: int = 100, tol=1e-5, generalized: bool = False,
                 initialization: str = 'unregularized', c=None, l1_ratio=None, constrained=False, stochastic=True):
        super().__init__(max_iter=max_iter, tol=tol, generalized=generalized,
                         initialization=initialization)
        self.stochastic = stochastic
        self.constrained = constrained
        self.c = c
        self.l1_ratio = l1_ratio

    def check_params(self):
        if self.c is None:
            self.c = [0] * len(self.views)
        elif isinstance(self.c, (float, int)):
            self.c = [self.c] * len(self.views)
        if self.l1_ratio is None:
            self.l1_ratio = [0] * len(self.views)
        elif isinstance(self.l1_ratio, (float, int)):
            self.l1_ratio = [self.l1_ratio] * len(self.views)
        if self.constrained:
            self.bin_init = np.zeros(len(self.views))

    def update_view(self, view_index: int):
        """
        :param view_index: index of view being updated
        :return: updated weights
        """
        if self.generalized:
            target = self.scores.mean(axis=0)
        else:
            target = self.scores[view_index - 1]
        if self.constrained:
            w, self.bin_init[view_index] = self.elastic_solver_constrained(self.views[view_index], target,
                                                                           alpha=self.c[view_index] / len(self.views),
                                                                           l1_ratio=self.l1_ratio[view_index],
                                                                           init=self.bin_init[view_index])
        else:
            w = self.elastic_solver(self.views[view_index], target,
                                    alpha=self.c[view_index] / len(self.views),
                                    l1_ratio=self.l1_ratio[view_index])
        assert (np.linalg.norm(w) > 0), 'all weights zero. try less regularisation or another initialisation'
        self.weights[view_index] = w / np.linalg.norm(self.views[view_index] @ w)
        self.scores[view_index] = (self.views[view_index] @ self.weights[view_index]).ravel()
        return w

    @ignore_warnings(category=ConvergenceWarning)
    def elastic_solver(self, X, y, alpha=0.1, l1_ratio=0.5):
        if self.stochastic:
            if l1_ratio == 0:
                beta = SGDRegressor(penalty='l2', alpha=alpha, fit_intercept=False, max_iter=10).fit(X, y.ravel()).coef_
            elif l1_ratio == 1:
                beta = SGDRegressor(penalty='l1', alpha=alpha, fit_intercept=False, max_iter=10).fit(X, y.ravel()).coef_
            else:
                beta = SGDRegressor(penalty='elasticnet', alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False,
                                    max_iter=10).fit(X, y.ravel()).coef_
        else:
            if alpha == 0:
                beta = LinearRegression(fit_intercept=False).fit(X, y.ravel()).coef_
            elif l1_ratio == 0:
                beta = Ridge(alpha=alpha, fit_intercept=False).fit(X, y.ravel()).coef_
            elif l1_ratio == 1:
                beta = Lasso(alpha=alpha, fit_intercept=False).fit(X, y.ravel()).coef_
            else:
                beta = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False).fit(X, y.ravel()).coef_
        return beta

    @ignore_warnings(category=ConvergenceWarning)
    def elastic_solver_constrained(self, X, y, alpha=0.1, l1_ratio=0.5, init=0):
        converged = False
        min_ = -1
        max_ = 1
        current = init
        previous = current
        previous_val = None
        i = 0
        while not converged:
            i += 1
            if self.stochastic:
                coef = SGDRegressor(penalty='elasticnet', alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False).fit(
                    np.sqrt(current + 1) * X,
                    y.ravel() / np.sqrt(
                        current + 1)).coef_
            else:
                coef = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False).fit(np.sqrt(current + 1) * X,
                                                                                           y.ravel() / np.sqrt(
                                                                                               current + 1)).coef_
            current_val = 1 - np.linalg.norm(X @ coef)
            current, previous, min_, max_ = _bin_search(current, previous, current_val, previous_val, min_, max_)
            previous_val = current_val
            if np.abs(current_val) < 1e-5:
                converged = True
            elif np.abs(max_ - min_) < 1e-30 or i == 50:
                converged = True
                coef = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False).fit(np.sqrt(current + 1) * X,
                                                                                           y / np.sqrt(
                                                                                               current + 1)).coef_
        return coef, current

    def objective(self):
        return elastic_cca_objective(self)


class ADMMInnerLoop(_InnerLoop):
    def __init__(self, max_iter: int = 100, tol=1e-5, generalized: bool = False,
                 initialization: str = 'unregularized', mu=None, lam=None, c=None, eta=None):
        super().__init__(max_iter=max_iter, tol=tol, generalized=generalized,
                         initialization=initialization)
        self.c = c
        self.lam = lam
        self.mu = mu
        self.eta = eta

    def check_params(self):
        if self.c is None:
            self.c = [0] * len(self.views)
        if self.lam is None:
            self.lam = [1] * len(self.views)
        if self.mu is None:
            self.mu = [lam / np.linalg.norm(view) ** 2 for lam, view in zip(self.lam, self.views)]
        if self.eta is None:
            self.eta = [0] * len(self.views)
        assert (all([mu > 0 for mu in self.mu])), "at least one mu is less than zero"
        assert (all([mu <= lam / np.linalg.norm(view) ** 2 for mu, lam, view in
                     zip(self.mu, self.lam,
                         self.views)])), "Condition from Parikh 2014 mu<lam/frobenius(X)**2"
        self.eta = [np.ones((view.shape[0], 1)) * eta for view, eta in zip(self.views, self.eta)]
        self.z = [np.zeros((view.shape[0], 1)) for view in self.views]
        self.l1_ratio = [1] * len(self.views)

    def update_view(self, view_index: int):
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
            self.weights[view_index] = self.prox_mu_f(self.weights[view_index] - mu / lam * self.views[view_index].T @ (
                    self.views[view_index] @ self.weights[view_index] - self.z[view_index] + self.eta[view_index]),
                                                      mu,
                                                      gradient, N * self.c[view_index])
            unnorm_z.append(np.linalg.norm(self.views[view_index] @ self.weights[view_index] + self.eta[view_index]))
            self.z[view_index] = self.prox_lam_g(
                self.views[view_index] @ self.weights[view_index] + self.eta[view_index])
            self.eta[view_index] = self.eta[view_index] + self.views[view_index] @ self.weights[view_index] - self.z[
                view_index]
            if np.linalg.norm(self.eta[view_index]) < 0.00000001:
                print('here')
            norm_eta.append(np.linalg.norm(self.eta[view_index]))
            norm_proj.append(np.linalg.norm(self.views[view_index] @ self.weights[view_index]))
            norm_weights.append(np.linalg.norm(self.weights[view_index], 1))
        assert (np.linalg.norm(
            self.weights[view_index]) > 0), 'all weights zero. try less regularisation or another initialisation'

        self.scores[view_index] = self.views[view_index] @ self.weights[view_index]

    def objective(self):
        return elastic_cca_objective(self)

    def prox_mu_f(self, x, mu, c, tau):
        u_update = x.copy()
        mask_1 = (x + (mu * c) > mu * tau)
        # if mask_1.sum()>0:
        u_update[mask_1] = x[mask_1] + mu * (c[mask_1] - tau)
        mask_2 = (x + (mu * c) < - mu * tau)
        # if mask_2.sum() > 0:
        u_update[mask_2] = x[mask_2] + mu * (c[mask_2] + tau)
        mask_3 = ~(mask_1 | mask_2)
        u_update[mask_3] = 0
        return u_update

    def prox_lam_g(self, x):
        norm = np.linalg.norm(x)
        if norm < 1:
            return x
        else:
            return x / max(1, norm)


def elastic_cca_objective(loop: _InnerLoop):
    """
    General objective function for sparse CCA |X_1w_1-X_2w_2|_2^2 + c_1|w_1|_1 + c_2|w_2|_1
    :param loop: an inner loop
    :return:
    """
    views = len(loop.views)
    c = np.array(loop.c)
    ratio = np.array(loop.l1_ratio)
    l1 = c * ratio
    l2 = c * (1 - ratio)
    total_objective = 0
    for i in range(views):
        # TODO this looks like it could be tidied up. In particular can we make the generalized objective correspond to the 2 view
        target = loop.scores.mean(axis=0)
        objective = views * np.linalg.norm(loop.views[i] @ loop.weights[i] - target) ** 2 / (2 * loop.views[i].shape[0])
        l1_pen = l1[i] * np.linalg.norm(loop.weights[i], ord=1)
        l2_pen = l2[i] * np.linalg.norm(loop.weights[i], ord=2)
        total_objective += objective + l1_pen + l2_pen
    return total_objective


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


def _soft_threshold(x, threshold):
    """
    if absolute value of x less than threshold replace with zero
    :param x: input
    :return: x soft-thresholded by threshold
    """
    diff = abs(x) - threshold
    diff[diff < 0] = 0
    out = np.sign(x) * diff
    return out


def _cosine_similarity(a, b):
    """
    Calculates the cosine similarity between vectors
    :param a: 1d numpy array
    :param b: 1d numpy array
    :return: cosine similarity
    """
    # https: // www.statology.org / cosine - similarity - python /
    return a.T @ b / (np.linalg.norm(a) * np.linalg.norm(b))
