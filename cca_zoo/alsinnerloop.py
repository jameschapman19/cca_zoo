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
                 initialization: str = 'random', params=None,
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
        if method in ['scca']:
            self.params['l1_ratio'] = [1 for _ in views]
        self.method = method
        if len(views) > 2:
            self.generalized = True
        else:
            self.generalized = generalized
        self.datasets = list(views)
        self.track_lyuponov = []
        self.track_correlation = []
        self.lyuponov = self.cca_lyuponov
        self.iterate()

    def iterate(self):
        # Weight vectors for y (normalized to 1)
        self.weights = [np.random.rand(dataset.shape[1]) for dataset in self.datasets]
        # initialize with first column
        if self.initialization == 'random':
            self.scores = np.array([np.random.rand(dataset.shape[0]) for dataset in self.datasets])
        elif self.initialization == 'first_column':
            self.scores = np.array([dataset[:, 0] / np.linalg.norm(dataset[:, 0]) for dataset in self.datasets])
        elif self.initialization == 'unregularized':
            self.scores = AlsInnerLoop(*self.datasets, initialization='random').scores

        self.bin_search_init = np.zeros(len(self.datasets))
        # select update function: needs to return new weights and update the target matrix as appropriate
        # might deprecate l2 and push it through elastic instead
        if self.method == 'pmd':
            self.update_function = self.pmd_update
            self.lyuponov = self.pls_lyuponov
        elif self.method == 'parkhomenko':
            self.update_function = self.parkhomenko_update
        elif self.method == 'elastic':
            self.update_function = self.elastic_update
            self.inverses = [pinv2(dataset) if dataset.shape[0] > dataset.shape[1] else None for dataset in
                             self.datasets]
        elif self.method == 'scca':
            self.update_function = self.scca_update
            self.inverses = [pinv2(dataset) if dataset.shape[0] > dataset.shape[1] else None for dataset in
                             self.datasets]
        else:
            self.update_function = self.method

        # This loops through each view and udpates both the weights and targets where relevant
        for _ in range(self.max_iter):
            for i, view in enumerate(self.datasets):
                self.weights[i] = self.update_function(i)

            self.track_lyuponov.append(self.lyuponov())

            # Some kind of early stopping
            if _ > 0 and all(cosine_similarity(self.weights[n], self.old_weights[n]) > (1 - self.tol) for n, view in
                             enumerate(self.scores)):
                break

            self.old_weights = self.weights.copy()
            # Sum all pairs
            self.track_correlation.append(np.corrcoef(self.scores)[np.triu_indices(self.scores.shape[0], 1)].sum())
        return self

    def pmd_update(self, view_index: int):
        """
        :param view_index: index of view being updated
        :return: updated weights
        """
        targets = np.ma.array(self.scores, mask=False)
        targets.mask[view_index] = True
        w = self.datasets[view_index].T @ targets.sum(axis=0).filled()
        w, w_success = self.delta_search(w, self.params['c'][view_index])
        if not w_success:
            w = self.datasets[view_index].T @ targets.sum(axis=0).filled()
        w /= np.linalg.norm(w)
        self.scores[view_index] = self.datasets[view_index] @ w
        return w

    def parkhomenko_update(self, view_index: int):
        """
        :param view_index: index of view being updated
        :return: updated weights
        """
        targets = np.ma.array(self.scores, mask=False)
        targets.mask[view_index] = True
        w = self.datasets[view_index].T @ targets.sum(axis=0).filled()
        if np.linalg.norm(w) == 0:
            w = self.datasets[view_index].T @ targets.sum().filled()
        w /= np.linalg.norm(w)
        w = self.soft_threshold(w, self.params['c'][view_index] / 2)
        if np.linalg.norm(w) == 0:
            w = self.datasets[view_index].T @ targets.sum()
        w /= np.linalg.norm(w)
        self.scores[view_index] = self.datasets[view_index] @ w
        return w

    def elastic_update(self, view_index: int):
        """
        :param view_index: index of view being updated
        :return: updated weights
        """
        if self.generalized:
            target = self.scores.mean(axis=0)
            w = self.elastic_solver(self.datasets[view_index], target,
                                    alpha=self.params['c'][view_index] / len(self.datasets),
                                    l1_ratio=self.params['l1_ratio'][view_index])
        else:
            w = self.elastic_solver(self.datasets[view_index], self.scores[view_index - 1], self.inverses[view_index],
                                    alpha=self.params['c'][view_index],
                                    l1_ratio=self.params['l1_ratio'][view_index])
        w /= np.linalg.norm(self.datasets[view_index] @ w)
        self.scores[view_index] = self.datasets[view_index] @ w
        return w

    def scca_update(self, view_index: int):
        """
        :param view_index: index of view being updated
        :return: updated weights
        """
        if self.generalized:
            target = self.scores.mean(axis=0)
            w = self.lasso_solver(self.datasets[view_index], target, self.inverses[view_index],
                                  alpha=self.params['c'][view_index] / len(self.datasets))
        else:
            w = self.lasso_solver(self.datasets[view_index], self.scores[view_index - 1], self.inverses[view_index],
                                  alpha=self.params['c'][view_index])
        if np.linalg.norm(self.datasets[view_index] @ w) == 0:
            print('failed')
            w = np.random.rand(*w.shape)
        w /= np.linalg.norm(self.datasets[view_index] @ w)
        self.scores[view_index] = self.datasets[view_index] @ w
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
            if not np.any(beta):
                beta = np.ones(beta.shape)
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

    def cca_lyuponov(self):
        views = len(self.datasets)
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
            lyuponov += 1 / (2 * self.datasets[i].shape[0]) * multiplier * np.linalg.norm(
                self.datasets[i] @ self.weights[i] - lyuponov_target) ** 2 + l1[i] * np.linalg.norm(self.weights[i],
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
