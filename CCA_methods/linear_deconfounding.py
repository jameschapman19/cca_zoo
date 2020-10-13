import itertools

from scipy.linalg import pinv2, block_diag, cholesky
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.testing import ignore_warnings

from CCA_methods.generate_data import *
from CCA_methods.plot_utils import cv_plot


class Wrapper:
    # The idea is that this can take some data and run one of my many CCA_archive methods
    def __init__(self, outdim_size=2, method='l2', generalized=True, max_iter=50, tol=1e-5):
        self.outdim_size = outdim_size
        self.method = method
        self.generalized = generalized
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, *args, gamma=None, params=None):

        self.params = params

        if params is None:
            self.params = {}
        if self.method == 'l2':
            if params is None:
                self.params = {'c': [0] * len(args)}

        # Fit returns in-sample score vectors and correlations as well as models with transform functionality
        self.dataset_list = []
        self.dataset_means = []
        for dataset in args:
            self.dataset_means.append(dataset.mean(axis=0))
            self.dataset_list.append(dataset - dataset.mean(axis=0))

        if self.method == 'mcca':
            assert all([view.shape[1] <= view.shape[0] for view in self.dataset_list])
            self.fit_mcca(*self.dataset_list, gamma=gamma)
        elif self.method == 'gcca':
            assert all([view.shape[1] <= view.shape[0] for view in self.dataset_list])
            self.fit_gcca(*self.dataset_list, gamma=gamma)
        else:
            self.outer_loop(*self.dataset_list, gamma)
            # have to do list comphrehension due to different dimensions in views
            if self.method[:4] == 'tree':
                self.tree_list = [self.tree_list[i] for i in range(len(args))]
                self.weights_list = [np.expand_dims(tree.feature_importances_, axis=1) for tree in self.tree_list]
            else:
                self.rotation_list = [
                    self.weights_list[i] @ pinv2(self.loading_list[i].T @ self.weights_list[i], check_finite=False) for
                    i in
                    range(len(args))]

        self.train_correlations = self.predict_corr(*args)
        return self

    def cv_fit(self, *args, param_candidates, folds=5, verbose=False):
        self.params.update(
            cross_validate(*args, max_iter=self.max_iter, outdim_size=self.outdim_size, method=self.method,
                           param_candidates=param_candidates, folds=folds,
                           verbose=verbose))
        self.fit(*args)
        return self

    def predict_corr(self, *args):
        # Takes two datasets and predicts their out of sample correlation using trained model
        transformed_views = self.transform_view(*args)
        all_corrs = []
        for x, y in itertools.product(transformed_views, repeat=2):
            all_corrs.append(np.diag(np.corrcoef(x.T, y.T)[:self.outdim_size, self.outdim_size:]))
        all_corrs = np.array(all_corrs).reshape((len(args), len(args), self.outdim_size))
        return all_corrs

    def predict_view(self, *args):
        # Regress original given views onto target
        transformed_views = self.transform_view(*args)

        # Get the regression from the training data with available views
        predicted_target = np.mean([transformed_views[i] for i in range(len(args)) if args[i] is not None], axis=0)

        predicted_views = []
        for i, view in enumerate(args):
            if view is None:
                predicted_views.append(predicted_target @ pinv2(self.weights_list[i]))
            else:
                predicted_views.append(view)
        for i, predicted_view in enumerate(predicted_views):
            predicted_views[i] += self.dataset_means[i]
        return predicted_views

    def transform_view(self, *args):
        # Demeaning
        new_views = []
        for i, new_view in enumerate(args):
            if new_view is None:
                new_views.append(None)
            else:
                new_views.append(new_view - self.dataset_means[i])

        if self.method == 'kernel':
            transformed_views = list(self.KCCA.transform(new_views[0], new_views[1]))
        elif self.method == 'pls':
            transformed_views = list(self.PLS.transform(new_views[0], new_views[1]))
        elif self.method[:4] == 'tree':
            transformed_views = []
            for i, new_view in enumerate(new_views):
                if new_view is None:
                    transformed_views.append(None)
                else:
                    transformed_views.append(self.tree_list[i].predict(new_view))
        else:
            transformed_views = []
            for i, new_view in enumerate(new_views):
                if new_view is None:
                    transformed_views.append(None)
                else:
                    transformed_views.append(new_view @ self.rotation_list[i])
        # d x n x k
        return transformed_views

    def outer_loop(self, *args):
        # list of d: p x k
        self.weights_list = [np.zeros((args[i].shape[1], self.outdim_size)) for i in range(len(args))]
        # list of d: n x k
        self.score_list = [np.zeros((args[i].shape[0], self.outdim_size)) for i in range(len(args))]
        # list of d:
        self.loading_list = [np.zeros((args[i].shape[1], self.outdim_size)) for i in range(len(args))]

        if len(args) == 2:
            C_train = args[0].T @ args[1]
            C_train_res = C_train.copy()
        else:
            C_train_res = None

        residuals = list(args)
        # For each of the dimensions
        for k in range(self.outdim_size):
            self.inner_loop = ALS_inner_loop(*residuals, C=C_train_res, generalized=self.generalized,
                                             params=self.params,
                                             method=self.method, max_iter=self.max_iter)
            for i in range(len(args)):
                if self.method[:4] == 'tree':
                    self.tree_list = self.inner_loop.weights
                else:
                    self.weights_list[i][:, k] = self.inner_loop.weights[i]
                    self.score_list[i][:, k] = self.inner_loop.targets[i, :]
                    self.loading_list[i][:, k] = residuals[i].T @ self.score_list[i][:, k] / np.linalg.norm(
                        self.score_list[i][:, k])
                    residuals[i] -= np.outer(self.score_list[i][:, k] / np.linalg.norm(self.score_list[i][:, k]),
                                             self.loading_list[i][:, k])

        return self

    def fit_mcca(self, *args, gamma=None):

        all_views = np.concatenate(args, axis=1)
        C = all_views.T @ all_views

        # Can regularise by adding to diagonal
        D = block_diag(*[m.T @ m for m in args])
        D[np.diag_indices_from(D)] = D.diagonal() + self.params['c'][0]

        C -= D
        R = cholesky(D, lower=False)

        whitened = np.linalg.inv(R.T) @ C @ np.linalg.inv(R)

        [eigvals, eigvecs] = np.linalg.eig(whitened)
        idx = np.argsort(eigvals, axis=0)[::-1]
        eigvecs = eigvecs[:, idx].real
        eigvals = eigvals[idx].real

        # sum p_i * sum p_i
        eigvecs = np.linalg.inv(R) @ eigvecs

        splits = np.cumsum([0] + [view.shape[1] for view in args])
        self.weights_list = [eigvecs[splits[i]:splits[i + 1], :self.outdim_size] for i in range(len(args))]
        self.rotation_list = self.weights_list
        self.score_list = [self.dataset_list[i] @ self.weights_list[i] for i in range(len(args))]

    def fit_gcca(self, *args, gamma=None):
        Q = []
        if gamma is None:
            gamma = [1] * len(args)
        for i, view in enumerate(args):
            Q.append(gamma[i] * view @ np.linalg.inv(view.T @ view) @ view.T)
        Q = np.sum(Q, axis=0)
        Q[np.diag_indices_from(Q)] = Q.diagonal() + self.params['c'][0]

        [eigvals, eigvecs] = np.linalg.eig(Q)
        idx = np.argsort(eigvals, axis=0)[::-1]
        eigvecs = eigvecs[:, idx].real
        eigvals = eigvals[idx].real

        self.weights_list = [np.linalg.inv(view.T @ view) @ view.T @ eigvecs[:, :self.outdim_size] for view in args]
        self.rotation_list = self.weights_list
        self.score_list = [self.dataset_list[i] @ self.weights_list[i] for i in range(len(args))]


class ALS_inner_loop:
    # an alternating least squares inner loop
    def __init__(self, *args, C=None, max_iter=100, tol=1e-5, generalized=True, initialization='random', params=None,
                 method='l2'):
        self.initialization = initialization
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.params = params
        if params is None:
            self.params = {'c': [0.001 for _ in args]}
        self.method = method
        self.generalized = generalized
        self.datasets = list(args)
        self.track_lyuponov = []
        self.track_correlation = []
        self.iterate()

    def iterate(self):
        # Weight vectors for y (normalized to 1)
        self.weights = [np.random.rand(dataset.shape[1]) for dataset in self.datasets]

        self.old_weights = [np.random.rand(dataset.shape[1]) for dataset in self.datasets]
        # initialize with first column
        self.targets = np.array([dataset[:, 0] / np.linalg.norm(dataset[:, 0]) for dataset in self.datasets])

        self.inverses = [pinv2(dataset) if dataset.shape[0] > dataset.shape[1] else None for dataset in self.datasets]
        self.corrs = []
        self.bin_search_init = np.zeros(len(self.datasets))

        # select update function: needs to return new weights and update the target matrix as appropriate
        if self.method == 'l2':
            self.update_function = self.ridge_update
        elif self.method == 'pmd':
            self.update_function = self.pmd_update
        elif self.method == 'parkhomenko':
            self.update_function = self.parkhomenko_update
        elif self.method == 'elastic':
            self.update_function = self.elastic_update
        elif self.method == 'scca':
            self.update_function = self.sparse_update
        elif self.method == 'elastic_jc':
            self.update_function = self.elastic_jc_update
        elif self.method == 'tree_jc':
            self.update_function = self.tree_update
        elif self.method == 'tree_jc2':
            self.update_function = self.tree_update2
        elif self.method == 'constrained_scca':
            self.update_function = self.constrained_update

        # This loops through each view and udpates both the weights and targets where relevant
        for _ in range(self.max_iter):
            for i, view in enumerate(self.datasets):
                self.weights[i] = self.update_function(i)
            if self.method[:4] != 'tree':
                self.track_lyuponov.append(self.elastic_lyuponov())
            # Some kind of early stopping
            if _ > 0:
                if all(np.linalg.norm(self.targets[n] - self.old_targets[n]) < self.tol for n, view in
                       enumerate(self.targets)):
                    break
                # if all(np.linalg.norm(self.weights[n] - self.old_weights[n]) < self.tol for n, view in
                # enumerate(self.datasets)):
                # break
            self.old_targets = self.targets.copy()
            # Sum all pairs
            self.corrs.append(np.corrcoef(self.targets)[np.triu_indices(self.targets.shape[0], 1)].sum())
        return self

    def ridge_update(self, view_index):
        if self.generalized:
            w = self.ridge_solver(self.datasets[view_index], self.targets.mean(axis=0), self.inverses[view_index],
                                  alpha=self.params['c'][view_index] / len(self.datasets))
        else:
            w = self.ridge_solver(self.datasets[view_index], self.targets[view_index - 1], self.inverses[view_index],
                                  alpha=self.params['c'][view_index])
        w /= np.linalg.norm(self.datasets[view_index] @ w)
        self.targets[view_index] = self.datasets[view_index] @ w
        return w

    def pmd_update(self, view_index):
        targets = np.ma.array(self.targets, mask=False)
        targets.mask[view_index] = True
        w = self.datasets[view_index].T @ targets.sum(axis=0).filled()
        w, w_success = self.delta_search(w, self.params['c'][view_index])
        if not w_success:
            w = self.datasets[view_index].T @ targets.sum(axis=0).filled()
            if np.linalg.norm(w) == 0:
                print('here')
        self.targets[view_index] = self.datasets[view_index] @ w
        return w

    def parkhomenko_update(self, view_index):
        targets = np.ma.array(self.targets, mask=False)
        targets.mask[view_index] = True
        w = self.datasets[view_index].T @ targets.sum().filled()
        if np.linalg.norm(w) == 0:
            w = self.datasets[view_index].T @ targets.sum().filled()
        w /= np.linalg.norm(w)
        w = self.soft_threshold(w, self.params['c'][view_index] / 2)
        if np.linalg.norm(w) == 0:
            w = self.datasets[view_index].T @ targets.sum()
        w /= np.linalg.norm(w)
        self.targets[view_index] = self.datasets[view_index] @ w
        return w

    def elastic_update(self, view_index):
        if self.generalized:
            w = self.elastic_solver(self.datasets[view_index], self.targets.mean(axis=0),
                                    alpha=self.params['c'][view_index] / len(self.datasets),
                                    l1_ratio=self.params['ratio'][view_index])
        else:
            w = self.elastic_solver(self.datasets[view_index], self.targets[view_index - 1],
                                    alpha=self.params['c'][view_index],
                                    l1_ratio=self.params['ratio'][view_index])
        w /= np.linalg.norm(self.datasets[view_index] @ w)
        self.targets[view_index] = self.datasets[view_index] @ w
        return w

    def sparse_update(self, view_index):
        if self.generalized:
            w = self.lasso_solver(self.datasets[view_index], self.targets.mean(axis=0), self.inverses[view_index],
                                  alpha=self.params['c'][view_index] / len(self.datasets))
        else:
            w = self.lasso_solver(self.datasets[view_index], self.targets[view_index - 1], self.inverses[view_index],
                                  alpha=self.params['c'][view_index])
        if np.linalg.norm(self.datasets[view_index] @ w) == 0:
            print('here')
        w /= np.linalg.norm(self.datasets[view_index] @ w)
        self.targets[view_index] = self.datasets[view_index] @ w
        return w

    def elastic_jc_update(self, view_index):
        if self.generalized:
            w, self.bin_search_init[view_index] = self.constrained_elastic(self.datasets[view_index],
                                                                           self.targets.mean(axis=0),
                                                                           self.params['c'][view_index] / len(
                                                                               self.datasets),
                                                                           self.params['ratio'][view_index],
                                                                           init=self.bin_search_init[view_index])
        else:
            w, self.bin_search_init[view_index] = self.constrained_elastic(self.datasets[view_index],
                                                                           self.targets[view_index - 1],
                                                                           self.params['c'][view_index],
                                                                           self.params['ratio'][view_index],
                                                                           init=self.bin_search_init[view_index])
        self.targets[view_index] = self.datasets[view_index] @ w
        return w

    def tree_update(self, view_index):
        if self.generalized:
            w, self.bin_search_init[view_index] = self.constrained_tree(self.datasets[view_index],
                                                                        self.targets.mean(axis=0),
                                                                        self.params['c'][view_index],
                                                                        init=self.bin_search_init[view_index])
        else:
            w, self.bin_search_init[view_index] = self.constrained_tree(self.datasets[view_index],
                                                                        self.targets[view_index - 1],
                                                                        self.params['c'][view_index],
                                                                        init=self.bin_search_init[view_index])
        self.targets[view_index] = w.predict(self.datasets[view_index])
        return w

    def tree_update2(self, view_index):
        if self.generalized:
            w = DecisionTreeRegressor(max_depth=self.params['c'][view_index]).fit(self.datasets[view_index],
                                                                                  self.targets.mean(axis=0))

        else:
            w = DecisionTreeRegressor(max_depth=self.params['c'][view_index]).fit(self.datasets[view_index],
                                                                                  self.targets[view_index - 1])

        self.targets[view_index] = w.predict(self.datasets[view_index])
        self.targets[view_index] /= np.linalg.norm(self.targets[view_index])
        return w

    def constrained_update(self, view_index):
        if self.generalized:
            w, self.bin_search_init[view_index] = self.constrained_regression(self.datasets[view_index],
                                                                              self.targets.mean(axis=0),
                                                                              self.params['c'][view_index],
                                                                              init=self.bin_search_init[view_index])
        else:
            w, self.bin_search_init[view_index] = self.constrained_regression(self.datasets[view_index],
                                                                              self.targets[view_index - 1],
                                                                              self.params['c'][view_index],
                                                                              init=self.bin_search_init[view_index])
        self.targets[view_index] = self.datasets[view_index] @ w
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
    def ridge_solver(self, X, y, X_inv=None, alpha=0.1):
        if alpha == 0:
            if X_inv is not None:
                beta = np.dot(X_inv, y)
            else:
                clf = LinearRegression(fit_intercept=False)
                clf.fit(X, y)
                beta = clf.coef_
        else:
            clf = Ridge(alpha=alpha, fit_intercept=False)
            clf.fit(X, y)
            beta = clf.coef_
        return beta

    @ignore_warnings(category=ConvergenceWarning)
    def elastic_solver(self, X, y, alpha=0.1, l1_ratio=0.5):
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

        if alpha == 0:
            coef = LinearRegression(fit_intercept=False).fit(X, y).coef_
        converged = False
        min_ = -1
        max_ = 10
        current = init
        previous = current
        previous_val = None
        i = 0
        while not converged:
            i += 1
            # coef = Lasso(alpha=current, selection='cyclic', max_iter=10000).fit(X, y).coef_
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

    @ignore_warnings(category=ConvergenceWarning)
    def constrained_regression(self, X, y, p, init=1):
        if p == 0:
            coef = LinearRegression(fit_intercept=False).fit(X, y).coef_
        converged = False
        min_ = 0
        max_ = 1
        current = init
        previous = current
        previous_val = None
        i = 0
        while not converged:
            i += 1
            # coef = Lasso(alpha=current, selection='cyclic', max_iter=10000).fit(X, y).coef_
            coef = Lasso(alpha=current, selection='cyclic', fit_intercept=False).fit(X, y).coef_
            if np.linalg.norm(X @ coef) > 0:
                current_val = p - np.linalg.norm(coef / np.linalg.norm(X @ coef), ord=1)
            else:
                current_val = p
            current, previous, min_, max_ = bin_search(current, previous, current_val, previous_val, min_, max_)
            previous_val = current_val
            if np.abs(current_val) < 1e-5:
                converged = True
            elif current < 1e-15:
                converged = True
            elif np.abs(max_ - min_) < 1e-30 or i == 50:
                converged = True
                coef = Lasso(alpha=min_, selection='cyclic', fit_intercept=False).fit(X, y).coef_
        coef = coef / np.linalg.norm(X @ coef)
        return coef, current

    def elastic_lyuponov(self):
        views = len(self.datasets)
        c = np.array(self.params.get('c', [0] * views))
        ratio = np.array(self.params.get('ratio', [1] * views))
        l1 = c * ratio
        l2 = c * (1 - ratio)
        lyuponov = 0
        for i in range(views):
            if self.generalized:
                lyuponov_target = self.targets.mean(axis=0)
                multiplier = views
            else:
                lyuponov_target = self.targets[i - 1]
                multiplier = 0.5
            lyuponov += 1 / (2 * self.datasets[i].shape[0]) * multiplier * np.linalg.norm(
                self.datasets[i] @ self.weights[i] - lyuponov_target) + l1[i] * np.linalg.norm(self.weights[i], ord=1) + \
                        l2[i] * np.linalg.norm(self.weights[i], ord=2)
        return lyuponov

    def elastic_lyuponov(self):
        views = len(self.datasets)
        c = np.array(self.params.get('c', [0] * views))
        ratio = np.array(self.params.get('ratio', [1] * views))
        l1 = c * ratio
        l2 = c * (1 - ratio)
        lyuponov = 0
        for i in range(views):
            if self.generalized:
                lyuponov_target = self.targets.mean(axis=0)
                multiplier = views
            else:
                lyuponov_target = self.targets[i - 1]
                multiplier = 0.5
            distance = np.linalg.norm(
                self.datasets[i] @ self.weights[i] - lyuponov_target)
            lyuponov += 1 / (2 * self.datasets[i].shape[0]) * multiplier * distance ** 2 + l1[i] * np.linalg.norm(
                self.weights[i], ord=1) + \
                        l2[i] * np.linalg.norm(self.weights[i], ord=2)
        return lyuponov


def permutation_test(train_set_1, train_set_2, outdim_size=5,
                     method='als', params=None, n_reps=100, level=0.05):
    if params is None:
        params = {}
    rho_train = np.zeros((n_reps, outdim_size))

    for _ in range(n_reps):
        print('permutation test rep: ', _ / n_reps, flush=True)
        results = Wrapper(outdim_size=outdim_size, method=method, params=params).fit(train_set_1,
                                                                                     train_set_2).train_correlations
        np.random.shuffle(train_set_1)
        rho_train[_, :] = results

    p_vals = np.zeros(outdim_size)
    # FWE Adjusted
    for i in range(outdim_size):
        p_vals[i] = (1 + (rho_train[:, 0] > rho_train[0, i]).sum()) / n_reps
    hypothesis_test = False
    significant_dims = 0
    while not hypothesis_test:
        if p_vals[significant_dims] > level:
            hypothesis_test = True
        else:
            significant_dims += 1
        if significant_dims == len(p_vals):
            hypothesis_test = True

    print('significant dims at level: ', str(level * 100), '%:', str(significant_dims), flush=True)
    print(p_vals, flush=True)
    return p_vals, significant_dims


def cross_validate(*args, max_iter=100, outdim_size=5, method='l2', param_candidates=None, folds=5,
                   verbose=False):
    print('cross validation with ', method, flush=True)
    print('number of folds: ', folds, flush=True)

    # Set up an array for each set of hyperparameters (perhaps could construct this automatically in the future?)
    assert (len(param_candidates) > 0)
    hyperparameter_grid_shape = [len(v) for k, v in param_candidates.items()]
    hyperparameter_scores = np.zeros(tuple([folds] + hyperparameter_grid_shape))

    # set up fold array. Suspect will need a function for this in future due t                                                                                                                      o family/twins etc.
    inds = np.arange(args[0].shape[0])
    np.random.shuffle(inds)
    if folds == 1:
        # If 1 fold do an 80:20 split
        fold_inds = np.array_split(inds, 5)
    else:
        fold_inds = np.array_split(inds, folds)

    for index, x in np.ndenumerate(hyperparameter_scores[0]):
        params = {}
        p_num = 0
        for key in param_candidates.keys():
            params[key] = param_candidates[key][index[p_num]]
            p_num += 1
        if verbose:
            print(params)
        for fold in range(folds):
            train_sets = [np.delete(data, fold_inds[fold], axis=0) for data in args]
            val_sets = [data[fold_inds[fold], :] for data in args]
            hyperparameter_scores[(fold,) + index] = \
                Wrapper(outdim_size=outdim_size, method=method, max_iter=max_iter).fit(
                    *train_sets, params=params).predict_corr(
                    *val_sets).sum(axis=-1)[np.triu_indices(len(args), 1)].sum()
        if verbose:
            print(hyperparameter_scores.sum(axis=0)[index] / folds)

    hyperparameter_scores_avg = hyperparameter_scores.sum(axis=0) / folds
    hyperparameter_scores_avg[np.isnan(hyperparameter_scores_avg)] = 0
    # Find index of maximum value from 2D numpy array
    result = np.where(hyperparameter_scores_avg == np.amax(hyperparameter_scores_avg))
    # Return the 1st
    best_params = {}
    p_num = 0
    for key in param_candidates.keys():
        best_params[key] = param_candidates[key][result[p_num][0].item()]
        p_num += 1
    print('Best score : ', np.amax(hyperparameter_scores_avg), flush=True)
    print(best_params, flush=True)
    if method == 'kernel':
        kernel_type = param_candidates.pop('kernel')[0]
        cv_plot(hyperparameter_scores_avg[0], param_candidates, method + ":" + kernel_type)
    elif not method == 'elastic':
        cv_plot(hyperparameter_scores_avg, param_candidates, method)
    return best_params


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


def main():
    # X1 = np.random.rand(100, 10)
    # X2 = np.random.rand(100, 11)
    # X3 = np.random.rand(100, 5)
    n = 100

    # X, Y, _, _ = generate_mai(n * 2, 1, 50, 50, sparse_variables_1=20,
    #                          sparse_variables_2=20, structure='toeplitz', sigma=0.95)

    X = np.random.rand(200, 20)
    Y = np.random.rand(200, 20)
    Z = np.random.rand(200, 20)

    x_test = X[:n, :]
    y_test = Y[:n, :]
    z_test = Z[:n, :]
    x_train = X[n:, :]
    y_train = Y[n:, :]
    z_train = Z[n:, :]

    x_test -= x_train.mean(axis=0)
    y_test -= y_train.mean(axis=0)
    z_test -= z_train.mean(axis=0)
    x_train -= x_train.mean(axis=0)
    y_train -= y_train.mean(axis=0)
    z_train -= z_train.mean(axis=0)
    params = {'c': [0, 0]}

    g_s = np.linspace(start=0,stop=10000000,num=100)
    for g in g_s:
        print(g)
        gcca = Wrapper(method='gcca', outdim_size=5, max_iter=1).fit(x_train, y_train, z_train, gamma=[1, 1, -g],
                                                                     params=params)
        print(gcca.train_correlations[2])

    l2 = Wrapper(method='l2', outdim_size=5, max_iter=1000).fit(x_train, y_train, params=params)

    mcca = Wrapper(method='mcca', outdim_size=5, max_iter=1).fit(x_train, y_train, params=params)

    c1 = [4, 5, 6]
    c2 = [4, 5, 6]
    param_candidates = {'c': list(itertools.product(c1, c2))}

    abc1 = Wrapper(method='tree_jc', outdim_size=1, max_iter=1).cv_fit(x_train, y_train,
                                                                       param_candidates=param_candidates, verbose=True)

    abc2 = Wrapper(method='tree_jc2', outdim_size=1, max_iter=1).cv_fit(x_train, y_train,
                                                                        param_candidates=param_candidates,
                                                                        verbose=True)

    params = {'c': [0.001, 0.001], 'ratio': [0.5, 0.5]}

    abc3 = Wrapper(method='elastic_jc', params=params, outdim_size=1, max_iter=100).fit(x_train, y_train)

    abc4 = Wrapper(method='elastic', params=params, outdim_size=1, max_iter=100).fit(x_train, y_train)
    bbb1 = abc1.predict_corr(x_test, y_test)
    bbb2 = abc2.predict_corr(x_test, y_test)
    bbb3 = abc3.predict_corr(x_test, y_test)
    print('here')


if __name__ == "__main__":
    main()
