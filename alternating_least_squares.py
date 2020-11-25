import numpy as np
from scipy.linalg import pinv2
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.testing import ignore_warnings


class ALS_inner_loop:
    """
    This is a wrapper class for alternating least squares based solutions to CCA
    """

    def __init__(self, *args, C=None, max_iter: int = 200, tol=1e-5, generalized: bool = False,
                 initialization: str = 'unregularized', params=None,
                 method: str = 'elastic', auxiliary: bool = True):
        self.initialization = initialization
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.params = params
        if params is None:
            self.params = {'c': [0 for _ in args], 'l1_ratio': [0 for _ in args]}
        if method in ['scca']:
            self.params['l1_ratio'] = [1 for _ in args]
        self.method = method
        if len(args) > 2:
            self.generalized = True
        else:
            self.generalized = generalized
        self.datasets = list(args)
        self.track_lyuponov = []
        self.track_correlation = []
        self.auxiliary = auxiliary
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
            self.scores = ALS_inner_loop(*self.datasets, initialization='random').scores

        if self.auxiliary:
            self.target = self.scores.mean(axis=0)

        self.inverses = [pinv2(dataset) if dataset.shape[0] > dataset.shape[1] else None for dataset in self.datasets]
        self.corrs = []
        self.bin_search_init = np.zeros(len(self.datasets))

        # select update function: needs to return new weights and update the target matrix as appropriate
        # might deprecate l2 and push it through elastic instead
        if self.method == 'pmd':
            self.update_function = self.pmd_update
        elif self.method == 'parkhomenko':
            self.update_function = self.parkhomenko_update
        elif self.method == 'elastic':
            self.update_function = self.elastic_update
        elif self.method == 'elastic_constrained':
            self.update_function = self.elastic_constrained_update
        elif self.method == 'scca':
            self.update_function = self.scca_update
        elif self.method == 'tree':
            self.update_function = self.tree_update
        elif self.method == 'no_normalize':
            self.update_function = self.no_normalize_update

        # This loops through each view and udpates both the weights and targets where relevant
        for _ in range(self.max_iter):
            for i, view in enumerate(self.datasets):
                if self.auxiliary:
                    self.target = self.scores.mean(axis=0)
                self.weights[i] = self.update_function(i)

            # tree doesn't have the lyuponov function
            if self.method[:4] != 'tree':
                if self.method == 'no_normalize':
                    self.track_lyuponov.append(self.lyuponov_exp())
                else:
                    self.track_lyuponov.append(self.lyuponov())
            # Some kind of early stopping
            if _ > 0:
                if all(np.linalg.norm(self.weights[n] - self.old_weights[n]) < self.tol for n, view in
                       enumerate(self.scores)):
                    break

            self.old_weights = self.weights.copy()
            # Sum all pairs
            self.corrs.append(np.corrcoef(self.scores)[np.triu_indices(self.scores.shape[0], 1)].sum())
        return self

    def pmd_update(self, view_index):
        targets = np.ma.array(self.scores, mask=False)
        targets.mask[view_index] = True
        w = self.datasets[view_index].T @ targets.sum(axis=0).filled()
        w, w_success = self.delta_search(w, self.params['c'][view_index])
        if not w_success:
            w = self.datasets[view_index].T @ targets.sum(axis=0).filled()
            if np.linalg.norm(w) == 0:
                print('failed')
        self.scores[view_index] = self.datasets[view_index] @ w
        return w

    def parkhomenko_update(self, view_index):
        targets = np.ma.array(self.scores, mask=False)
        targets.mask[view_index] = True
        w = self.datasets[view_index].T @ targets.sum().filled()
        if np.linalg.norm(w) == 0:
            w = self.datasets[view_index].T @ targets.sum().filled()
        w /= np.linalg.norm(w)
        w = self.soft_threshold(w, self.params['c'][view_index] / 2)
        if np.linalg.norm(w) == 0:
            w = self.datasets[view_index].T @ targets.sum()
        w /= np.linalg.norm(w)
        self.scores[view_index] = self.datasets[view_index] @ w
        return w

    def elastic_update(self, view_index):
        if self.generalized:
            if self.auxiliary:
                target = self.target
            else:
                target = self.scores.mean(axis=0)
            w = self.elastic_solver(self.datasets[view_index], target,
                                    alpha=self.params['c'][view_index] / len(self.datasets),
                                    l1_ratio=self.params['l1_ratio'][view_index])
        else:
            w = self.elastic_solver(self.datasets[view_index], self.scores[view_index - 1],
                                    alpha=self.params['c'][view_index],
                                    l1_ratio=self.params['l1_ratio'][view_index])
        w /= np.linalg.norm(self.datasets[view_index] @ w)
        self.scores[view_index] = self.datasets[view_index] @ w
        return w

    def scca_update(self, view_index):
        if self.generalized:
            if self.auxiliary:
                target = self.target
            else:
                target = self.scores.mean(axis=0)
            w = self.lasso_solver(self.datasets[view_index], target, self.inverses[view_index],
                                  alpha=self.params['c'][view_index] / len(self.datasets))
        else:
            w = self.lasso_solver(self.datasets[view_index], self.scores[view_index - 1], self.inverses[view_index],
                                  alpha=self.params['c'][view_index])
        if np.linalg.norm(self.datasets[view_index] @ w) == 0:
            print('failed')
        w /= np.linalg.norm(self.datasets[view_index] @ w)
        self.scores[view_index] = self.datasets[view_index] @ w
        return w

    def elastic_constrained_update(self, view_index):
        if self.generalized:
            if self.auxiliary:
                target = self.target
            else:
                target = self.scores.mean(axis=0)
            w, self.bin_search_init[view_index] = self.constrained_elastic(self.datasets[view_index],
                                                                           target,
                                                                           self.params['c'][view_index] / len(
                                                                               self.datasets),
                                                                           self.params['l1_ratio'][view_index],
                                                                           init=self.bin_search_init[view_index])
        else:
            w, self.bin_search_init[view_index] = self.constrained_elastic(self.datasets[view_index],
                                                                           self.scores[view_index - 1],
                                                                           self.params['c'][view_index],
                                                                           self.params['l1_ratio'][view_index],
                                                                           init=self.bin_search_init[view_index])
        self.scores[view_index] = self.datasets[view_index] @ w
        return w

    def tree_update(self, view_index):
        if self.generalized:
            if self.auxiliary:
                target = self.target
            else:
                target = self.scores.mean(axis=0)
        else:
            target = self.scores[view_index - 1]
        w, self.bin_search_init[view_index] = self.constrained_tree(self.datasets[view_index],
                                                                    target,
                                                                    depth=self.params['c'][view_index],
                                                                    init=self.bin_search_init[view_index])
        self.scores[view_index] = w.predict(self.datasets[view_index])
        return w

    def no_normalize_update(self, view_index):
        if self.generalized:
            if self.auxiliary:
                target = self.target
            else:
                target = self.scores.mean(axis=0)
            w = self.elastic_solver(self.datasets[view_index], target / np.linalg.norm(target),
                                    alpha=self.params['c'][view_index] / len(self.datasets),
                                    l1_ratio=self.params['l1_ratio'][view_index])
        self.scores[view_index] = self.datasets[view_index] @ w
        return w

    def soft_threshold(self, x, delta):
        diff = abs(x) - delta
        diff[diff < 0] = 0
        out = np.sign(x) * diff
        return out

    def bisec(self, K, c, x1, x2):
        # does a binary search between x1 and x2 (thresholds). Outputs weight vector.
        converge = False
        success = True
        tol = 1e-5
        while not converge and success:
            x = (x2 + x1) / 2
            out = self.soft_threshold(K, x)
            if np.linalg.norm(out, 2) > 0:
                out = out / np.linalg.norm(out, 2)
            else:
                out = np.empty(out.shape)
                out[:] = np.nan
            if np.sum(np.abs(out)) == 0:
                x2 = x
            elif np.linalg.norm(out, 1) > c:
                x1 = x
            elif np.linalg.norm(out, 1) < c:
                x2 = x

            diff = np.abs(np.linalg.norm(out, 1) - c)
            if diff <= tol:
                converge = True
            elif np.isnan(np.sum(diff)):
                success = False
        return out

    def delta_search(self, w, c):
        success = True
        delta = 0
        # update the weights
        # Do the soft thresholding operation
        up = self.soft_threshold(w, delta)
        if np.linalg.norm(up, 2) > 0:
            up = up / np.linalg.norm(up, 2)
        else:
            up = np.empty(up.shape)
            up[:] = np.nan

        # if the 1 norm of the weights is greater than c
        it = 0
        if np.linalg.norm(up, 1) > c:
            delta1 = delta
            delta2 = delta1 + 1.1
            converged = False
            max_delta = 0
            while not converged and success:
                up = self.soft_threshold(w, delta2)
                if np.linalg.norm(up, 2) > 0:
                    up = up / np.linalg.norm(up, 2)
                # if all zero or all nan then delta2 too big
                if np.sum(np.abs(up)) == 0 or np.isnan(np.sum(np.abs(up))):
                    delta2 = delta2 / 1.618
                # if too big then increase delta
                elif np.linalg.norm(up, 1) > c:
                    delta1 = delta2
                    delta2 = delta2 * 2
                # if there is slack then converged
                elif np.linalg.norm(up, 1) <= c:
                    converged = True

                # update the maximum attempted delta
                if delta2 > max_delta:
                    max_delta = delta2

                # if the threshold
                if delta2 == 0:
                    success = False
                it += 1
                if it == self.max_iter:
                    delta1 = 0
                    delta2 = max_delta
                    success = False
            if success:
                up = self.bisec(w, c, delta1, delta2)
                if np.isnan(np.sum(up)) or np.sum(up) == 0:
                    success = False
        return up, success

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
            lassoreg = Lasso(alpha=alpha, selection='cyclic', fit_intercept=False)
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
    def constrained_tree(self, X, y, depth=2, init=0):
        converged = False
        min_ = -1
        max_ = 10
        current = init
        previous = current
        previous_val = None
        i = 0
        while not converged:
            i += 1
            tree = DecisionTreeRegressor(max_depth=depth).fit(np.sqrt(current + 1) * X, y / np.sqrt(current + 1))
            current_val = 1 - np.linalg.norm(tree.predict(X))
            current, previous, min_, max_ = bin_search(current, previous, current_val, previous_val, min_, max_)
            previous_val = current_val
            if np.abs(current_val) < 1e-5:
                converged = True
            elif np.abs(max_ - min_) < 1e-5 or i == 50:
                converged = True
                tree = DecisionTreeRegressor(max_depth=depth).fit(np.sqrt(min_ + 1) * X, y / np.sqrt(min_ + 1))
        return tree, current

    @ignore_warnings(category=ConvergenceWarning)
    def constrained_elastic(self, X, y, alpha=0.1, l1_ratio=0.5, init=0):
        converged = False
        min_ = -1
        max_ = 0
        current = init
        previous = current
        previous_val = None
        i = 0
        while not converged:
            i += 1
            coef = self.elastic_solver(np.sqrt(current + 1) * X, y / np.sqrt(current + 1), alpha=alpha,
                                       l1_ratio=l1_ratio)
            current_val = 1 - np.linalg.norm(X @ coef)
            current, previous, min_, max_ = bin_search(current, previous, current_val, previous_val, min_, max_)
            previous_val = current_val
            if np.abs(current_val) < 1e-5:
                converged = True
            elif np.abs(max_ - min_) < 1e-30 or i == 50:
                converged = True
                print('warning: failed to converge')
        return coef, previous

    def lyuponov(self):
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

    def lyuponov_exp(self):
        views = len(self.datasets)
        c = np.array(self.params.get('c', [0] * views))
        ratio = np.array(self.params.get('l1_ratio', [0] * views))
        l1 = c * ratio
        l2 = c * (1 - ratio)
        lyuponov = 0
        for i in range(views):
            lyuponov_target = self.scores.mean(axis=0)
            lyuponov_target /= np.linalg.norm(lyuponov_target)
            multiplier = 1
            lyuponov += 1 / (2 * self.datasets[i].shape[0]) * multiplier * np.linalg.norm(
                self.datasets[i] @ self.weights[i] - lyuponov_target) ** 2 + l1[i] * np.linalg.norm(self.weights[i],
                                                                                                    ord=1) + \
                        l2[i] * np.linalg.norm(self.weights[i], ord=2)
        return lyuponov


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
        if previous_val < 0:
            new = (current + previous) / 2
        if current < max_:
            max_ = current
    return new, current, min_, max_
