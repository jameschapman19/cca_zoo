from itertools import combinations

import numpy as np
from scipy.linalg import pinv2
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.utils._testing import ignore_warnings


class AlsInnerLoop:
    """
    This class implements solutions to regularized CCA and PLS by alternating least squares.
    """

    def __init__(self, *views, max_iter: int = 100, tol=1e-3, generalized: bool = False,
                 initialization: str = 'unregularized', params=None,
                 method: str = 'elastic'):
        """
        :param views: numpy arrays separated by comma e.g. fit(view_1,view_2,view_3, params=params)
        :param max_iter: maximum number of updates
        :param tol: stop after cosine similarity between iterations within tolerance
        :param generalized: use average of previous scores as regression target. Required for #views>2
        :param initialization: 'random' random initialisation, 'first column' use first columns of views, 'unregularized' use unregularized pls solution
        :param params: parameters required by a given method
        :param method: either a string for standard methods:
        'elastic' : Waaijenborg https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2367589/
        'pmd' : Witten https://web.stanford.edu/~hastie/Papers/PMD_Witten.pdf
        'parkhomenko' : Parkhomenko https://www.degruyter.com/view/journals/sagmb/8/1/article-sagmb.2009.8.1.1406.xml.xml#:~:text=Sparse%20Canonical%20Correlation%20Analysis%20with%20Application%20to%20Genomic,for%20Microarray%20Data%20Based%20on%20Sensitivity%20and%20Meta-Analysis
        'scca' : https://onlinelibrary.wiley.com/doi/abs/10.1111/biom.13043
        or an update function of similar form
        """
        self.initialization = initialization
        self.max_iter = max_iter
        self.tol = tol
        self.params = params
        if params is None:
            self.params = {'c': [0 for _ in views], 'l1_ratio': [0 for _ in views]}
        self.method = method
        if len(views) > 2:
            self.generalized = True
        else:
            self.generalized = generalized
        self.views = views
        self.track_lyuponov = []
        self.track_correlation = []
        self.lyuponov = self.cca_lyuponov
        self.iterate()

    def iterate(self):
        # Weight vectors for y (normalized to 1)
        self.weights = [np.random.rand(dataset.shape[1]) for dataset in self.views]
        # initialize with first column
        if self.initialization == 'random':
            self.scores = np.array([np.random.rand(dataset.shape[0]) for dataset in self.views])
        elif self.initialization == 'first_column':
            self.scores = np.array([dataset[:, 0] / np.linalg.norm(dataset[:, 0]) for dataset in self.views])
        elif self.initialization == 'unregularized':
            self.scores = AlsInnerLoop(*self.views, initialization='random').scores

        self.bin_search_init = np.zeros(len(self.views))
        # select update function: needs to return new weights and update the target matrix as appropriate
        # might deprecate l2 and push it through elastic instead
        if self.method == 'pmd':
            self.update_function = self.pmd_update
            self.lyuponov = self.pls_lyuponov
        elif self.method == 'parkhomenko':
            self.update_function = self.parkhomenko_update
        elif self.method == 'admm':
            assert (all([mu > 0 for mu in self.params['mu']])), "at least one mu is less than zero"
            assert (all([mu < lam / np.linalg.norm(view) for mu, lam, view in
                         zip(self.params['mu'], self.params['lam'], self.views)])), "Condition from Parikh 2014"
            self.update_function = self.admm_update
            self.eta = [np.zeros(z.shape) for z in self.scores]
            self.params['l1_ratio'] = [1 for _ in self.views]
            self.lyuponov = self.cca_lyuponov
        elif self.method == 'elastic':
            self.update_function = self.elastic_update
            self.inverses = [pinv2(dataset) if dataset.shape[0] > dataset.shape[1] else None for dataset in
                             self.views]
        elif self.method == 'scca':
            self.update_function = self.scca_update
            self.inverses = [pinv2(dataset) if dataset.shape[0] > dataset.shape[1] else None for dataset in
                             self.views]
            # This is only used to calculate lyuponov function convergence
            self.params['l1_ratio'] = [1 for _ in self.views]
        else:
            self.update_function = self.method

        # This loops through each view and udpates both the weights and targets where relevant
        for _ in range(self.max_iter):
            for i, view in enumerate(self.views):
                self.weights[i] = self.update_function(i)

            self.track_lyuponov.append(self.lyuponov())
            # Sum all pairwise correlations
            self.track_correlation.append(np.corrcoef(self.scores)[np.triu_indices(self.scores.shape[0], 1)].sum())
            # Some kind of early stopping
            if _ > 0 and all(cosine_similarity(self.weights[n], self.old_weights[n]) > (1 - self.tol) for n, view in
                             enumerate(self.scores)):
                break
            self.old_weights = self.weights.copy()

        return self

    def pmd_update(self, view_index: int):
        """
        :param view_index: index of view being updated
        :return: updated weights
        """
        targets = np.ma.array(self.scores, mask=False)
        targets.mask[view_index] = True
        w = self.views[view_index].T @ targets.sum(axis=0).filled()
        w, w_success = self.delta_search(w, self.params['c'][view_index])
        assert (np.linalg.norm(w) > 0), 'all weights zero. try less regularisation or another initialisation'
        w /= np.linalg.norm(w)
        self.scores[view_index] = self.views[view_index] @ w
        return w

    def parkhomenko_update(self, view_index: int):
        """
        :param view_index: index of view being updated
        :return: updated weights
        """
        targets = np.ma.array(self.scores, mask=False)
        targets.mask[view_index] = True
        w = self.views[view_index].T @ targets.sum(axis=0).filled()
        assert (np.linalg.norm(w) > 0), 'all weights zero. try less regularisation or another initialisation'
        w /= np.linalg.norm(w)
        w = self.soft_threshold(w, self.params['c'][view_index] / 2)
        assert (np.linalg.norm(w) > 0), 'all weights zero. try less regularisation or another initialisation'
        w /= np.linalg.norm(w)
        self.scores[view_index] = self.views[view_index] @ w
        return w

    def admm_update(self, view_index: int):
        targets = np.ma.array(self.scores, mask=False)
        targets.mask[view_index] = True
        # Suo uses parameter tau whereas we use parameter c to penalize the 1-norm of the weights.
        # Suo uses c to refer to the gradient where we now use gradient
        gradient = self.views[view_index].T @ targets.sum(axis=0).filled()
        # reset eta each loop?
        # self.eta[view_index][:] = 0
        mu = self.params['mu'][view_index]
        lam = self.params['lam'][view_index]
        N = self.views[view_index].shape[0]
        last_scores = self.scores[view_index].copy()
        for _ in range(self.max_iter):
            # We multiply 'c' by N in order to make regularisation match across the different sparse cca methods
            self.weights[view_index] = self.prox_mu_f(self.weights[view_index] - mu / lam * self.views[view_index].T @ (
                    self.views[view_index] @ self.weights[view_index] - self.scores[view_index] + self.eta[view_index]),
                                                      mu,
                                                      gradient, N * self.params['c'][view_index])
            self.scores[view_index] = self.prox_lam_g(
                self.views[view_index] @ self.weights[view_index] + self.eta[view_index])
            self.eta[view_index] = self.eta[view_index] + self.views[view_index] @ self.weights[view_index] - \
                                   self.scores[
                                       view_index]
            if np.abs(np.linalg.norm(self.scores) - 1) < self.tol:
                break
        b = np.linalg.norm(self.views[view_index] @ self.weights[view_index])
        assert (np.linalg.norm(
            self.weights[view_index]) > 0), 'all weights zero. try less regularisation or another initialisation'
        return self.weights[view_index]

    def elastic_update(self, view_index: int):
        """
        :param view_index: index of view being updated
        :return: updated weights
        """
        if self.generalized:
            target = self.scores.mean(axis=0)
            w = self.elastic_solver(self.views[view_index], target,
                                    alpha=self.params['c'][view_index] / len(self.views),
                                    l1_ratio=self.params['l1_ratio'][view_index])
        else:
            w = self.elastic_solver(self.views[view_index], self.scores[view_index - 1], self.inverses[view_index],
                                    alpha=self.params['c'][view_index],
                                    l1_ratio=self.params['l1_ratio'][view_index])
        assert (np.linalg.norm(w) > 0), 'all weights zero. try less regularisation or another initialisation'
        w /= np.linalg.norm(self.views[view_index] @ w)
        self.scores[view_index] = self.views[view_index] @ w
        return w

    def scca_update(self, view_index: int):
        """
        :param view_index: index of view being updated
        :return: updated weights
        """
        if self.generalized:
            target = self.scores.mean(axis=0)
            w = self.lasso_solver(self.views[view_index], target, self.inverses[view_index],
                                  alpha=self.params['c'][view_index] / len(self.views))
        else:
            w = self.lasso_solver(self.views[view_index], self.scores[view_index - 1], self.inverses[view_index],
                                  alpha=self.params['c'][view_index])
        assert (np.linalg.norm(w) > 0), 'all weights zero. try less regularisation or another initialisation'
        w /= np.linalg.norm(self.views[view_index] @ w)
        self.scores[view_index] = self.views[view_index] @ w
        return w

    @staticmethod
    def soft_threshold(x, threshold):
        """
        if absolute value of x less than threshold replace with zero
        :param x: input
        :param delta: threshold
        :return: x soft-thresholded by threshold
        """
        diff = abs(x) - threshold
        diff[diff < 0] = 0
        out = np.sign(x) * diff
        return out

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
            coef = self.soft_threshold(w, current)
            if np.linalg.norm(coef) > 0:
                coef /= np.linalg.norm(coef)
            current_val = c - np.linalg.norm(coef, 1)
            current, previous, min_, max_ = bin_search(current, previous, current_val, previous_val, min_, max_)
            previous_val = current_val
            if np.abs(current_val) < 1e-5 or np.abs(max_ - min_) < 1e-30 or i == 50:
                converged = True
        return coef, current

    @ignore_warnings(category=ConvergenceWarning)
    def lasso_solver(self, X, y, X_inv=None, alpha=0.1):
        if alpha == 0:
            if X_inv is not None:
                beta = np.dot(X_inv, y)
            else:
                clf = LinearRegression(fit_intercept=False)
                clf.fit(X, y)
                beta = clf.coef_
        else:
            lassoreg = Lasso(alpha=alpha, selection='random', fit_intercept=False)
            lassoreg.fit(X, y)
            beta = lassoreg.coef_
        return beta

    @ignore_warnings(category=ConvergenceWarning)
    def elastic_solver(self, X, y, X_inv=None, alpha=0.1, l1_ratio=0.5):
        if alpha == 0:
            if X_inv is not None:
                beta = np.dot(X_inv, y)
            else:
                clf = LinearRegression(fit_intercept=False)
                clf.fit(X, y)
                beta = clf.coef_
        elif l1_ratio == 0:
            clf = Ridge(alpha=alpha, fit_intercept=False)
            clf.fit(X, y)
            beta = clf.coef_
        else:
            clf = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False)
            clf.fit(X, y)
            beta = clf.coef_
        return beta

    @ignore_warnings(category=ConvergenceWarning)
    def constrained_elastic(self, X, y, alpha=0.1, l1_ratio=0.5, init=0):
        converged = False
        min_ = -1
        max_ = 10
        current = init
        previous = current
        previous_val = None
        i = 0
        while not converged:
            i += 1
            coef = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False).fit(np.sqrt(current + 1) * X,
                                                                                       y / np.sqrt(current + 1)).coef_
            current_val = 1 - np.linalg.norm(X @ coef)
            current, previous, min_, max_ = bin_search(current, previous, current_val, previous_val, min_, max_)
            previous_val = current_val
            if np.abs(current_val) < 1e-5:
                converged = True
            elif np.abs(max_ - min_) < 1e-30 or i == 50:
                converged = True
                coef = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False).fit(np.sqrt(current + 1) * X,
                                                                                           y / np.sqrt(
                                                                                               current + 1)).coef_
        return coef, current

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
        return x / max(1, norm)

    def cca_lyuponov(self):
        views = len(self.views)
        c = np.array(self.params.get('c', [0] * views))
        ratio = np.array(self.params.get('l1_ratio', [0] * views))
        l1 = c * ratio
        l2 = c * (1 - ratio)
        lyuponov = 0
        for i in range(views):
            if self.generalized:
                lyuponov_target = self.scores.mean(axis=0)
                multiplier = views
            else:
                lyuponov_target = self.scores[i - 1]
                multiplier = 0.5
            lyuponov += 1 / (2 * self.views[i].shape[0]) * multiplier * np.linalg.norm(
                self.views[i] @ self.weights[i] - lyuponov_target) ** 2 + l1[i] * np.linalg.norm(self.weights[i],
                                                                                                 ord=1) + \
                        l2[i] * np.linalg.norm(self.weights[i], ord=2)
        return lyuponov

    def pls_lyuponov(self):
        cov = 0
        for (score_i, score_j) in combinations(self.scores, 2):
            cov += score_i.T @ score_j
        return cov


def bin_search(current, previous, current_val, previous_val, min_, max_):
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


def cosine_similarity(a, b):
    # https: // www.statology.org / cosine - similarity - python /
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
