import matplotlib
import numpy as np
from CCA_methods import *
from CCA_methods.generate_data import generate_candola
from scipy.linalg import pinv
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import lars_path

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plt.style.use('ggplot')
np.random.seed(42)


def constrained_regression(X, y, p):
    if p == 0:
        coef_star = LinearRegression(fit_intercept=False).fit(X, y).coef_
    else:
        alphas, _, coefs = lars_path(X, y)
        coefs_l1 = np.linalg.norm(coefs, ord=1, axis=0)
        projs_l2 = np.linalg.norm(X @ coefs, ord=2, axis=0)
        projs_l2[projs_l2 == 0] = 1e-200
        l1_normalized_norms = coefs_l1 / projs_l2
        l1_normalized_norms[np.isnan(l1_normalized_norms)] = 0
        alpha_max = np.argmax(l1_normalized_norms > p)
        alpha_min = alpha_max - 1
        if alpha_min == -1:
            coef_star = LinearRegression(fit_intercept=False).fit(X, y).coef_
        else:
            new_vec_hat = X @ (coefs[:, alpha_max] - coefs[:, alpha_min])
            old_vec = X @ coefs[:, alpha_min]
            m = coefs_l1[alpha_max] - coefs_l1[alpha_min]
            c = coefs_l1[alpha_min]
            A = m ** 2 - p ** 2 * new_vec_hat.T @ new_vec_hat
            B = 2 * m * c - 2 * p ** 2 * new_vec_hat.T @ old_vec
            C = c ** 2 - p ** 2 * old_vec.T @ old_vec
            f_pos = (-B + np.sqrt(B ** 2 - 4 * A * C)) / (2 * A)
            coef_star = coefs[:, alpha_min] + f_pos * (coefs[:, alpha_max] - coefs[:, alpha_min])
    proj_star = X @ coef_star
    coef = coef_star / np.linalg.norm(proj_star)
    return coef


def elastic_search(X, y, l1_ratio, alpha, n_tries=100):
    # Grid search for lam_1 and lam_2. Hmm.
    success = False
    tries = 0
    tol = 1e-5
    mu = -1 + tol
    mu_old = mu
    mu_min = -1
    mu_max = 0
    coef = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False).fit(np.sqrt(1 + mu) * X,
                                                                               y / np.sqrt(1 + mu)).coef_
    length_old = np.linalg.norm(X @ coef)
    while not success and tries < n_tries:
        tries += 1
        coef = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False).fit(np.sqrt(1 + mu) * X,
                                                                                   y / np.sqrt(1 + mu)).coef_

        length = np.linalg.norm(X @ coef)

        if np.abs(length - 1) < tol:
            success = True
        if not success:
            mu, mu_old, mu_min, mu_max = bin_search(mu, mu_old, 1 - length, 1 - length_old, mu_min, mu_max)
            length_old = length
    return success, coef


class ALS_inner_loop:
    # an alternating least squares inner loop
    def __init__(self, X, Y, max_iter=100, tol=1e-5, initialization='random', params=None,
                 method='l2'):
        self.initialization = initialization
        self.X = X
        self.Y = Y
        self.C = X.T @ Y
        self.max_iter = max_iter
        self.tol = tol
        self.params = params
        if params is None:
            self.params = {'c_1': 0, 'c_2': 0}
        self.method = method
        self.iterate()

    def iterate(self):
        # Any initialization required

        # Weight vectors for y (normalized to 1)
        c = np.ones(self.Y.shape[1])
        c /= np.linalg.norm(c)
        w = np.ones(self.X.shape[1])
        w /= np.linalg.norm(w)
        w_old = w
        c_old = c

        # Score/Projection vector for Y (normalized to 1)
        self.v = self.Y[:, 3]
        #self.v = np.ones(self.X.shape[0])
        self.v = self.v / np.linalg.norm(self.v)
        v = self.v
        self.u = self.X @ w
        u = self.u
        target = self.v
        #if self.method == 'james' or self.method == 'james_constrained' or self.method == 'mai' or self.method == 'james_elastic' or self.method == 'john' or self.method == 'john_constrained':
        #    warm_start = ALS_inner_loop(self.X, self.Y)
        #    target = self.X @ warm_start.w + self.Y @ warm_start.c
        #    target /= np.linalg.norm(target)
        #    self.v = target


        # Pre calculate inverses. Used for unregularized CCA_archive
        X_inv = pinv(self.X)
        Y_inv = pinv(self.Y)

        # Object to store statistics for convergence analysis
        self.track_correlation = np.zeros(self.max_iter)
        self.track_correlation[:] = np.nan
        self.track_covariance = np.zeros(self.max_iter)
        self.track_covariance[:] = np.nan
        self.track_distance = np.zeros(self.max_iter)
        self.track_distance[:] = np.nan

        self.track_norms_w = np.zeros(self.max_iter)
        self.track_norms_w[:] = np.nan
        self.track_norms_c = np.zeros(self.max_iter)
        self.track_norms_c[:] = np.nan

        self.track_obj = np.zeros(self.max_iter)
        self.track_obj[:] = np.nan

        self.alt_obj = np.zeros(self.max_iter)
        self.alt_obj[:] = np.nan

        self.w_success = True
        self.c_success = True

        for _ in range(self.max_iter):

            # Update W
            ## I've implemented each update as I understand it for w and c. At the moment I've just repeated the updates for w and c
            if self.method == 'l2':
                # Use ridge solver or pseudo-inverse
                w = self.ridge_solver(self.X, v, X_inv, alpha=self.params['c_1'])
                w /= np.linalg.norm(self.X @ w)
            elif self.method == 'witten':
                # Use covariance matrix and then iterative soft-thresholding to fuflfil constraint on w directly
                w = self.C @ c
                w, w_success = self.update(w, self.params['c_1'])
                if not w_success:
                    w = self.C @ c
                    w /= np.linalg.norm(w)
            elif self.method == 'parkhomenko':
                # Use covariance matrix and then iterative soft-thresholding at defined level
                w = self.C @ c
                if np.linalg.norm(w) == 0:
                    w = self.C @ c
                w /= np.linalg.norm(w)
                w = self.soft_threshold(w, self.params['c_1'] / 2)
                if np.linalg.norm(w) == 0:
                    w = self.C @ c
                w /= np.linalg.norm(w)
            elif self.method == 'SAR':
                # Apply lasso solver directly
                w = self.lasso_solver(self.X, v, alpha=self.params['c_1'])
                w /= np.linalg.norm(w)
            elif self.method == 'waaijenborg':
                # Apply elastic net
                w = self.elastic_solver(self.X, v, alpha=self.params['c_1'], l1_ratio=self.params['l1_ratio_1'])
                w /= np.linalg.norm(X @ w)
            elif self.method == 'james':
                w = constrained_regression(self.X, v, self.params['c_1'])
            elif self.method == 'james_elastic':
                # w = constrained_elastic_regression(self.X, v, self.params['c_1'])
                self.w_success, w = elastic_search(self.X, v, 0.9, self.params['c_1'])
            elif self.method == 'john':
                w = self.lasso_solver(self.X, target, X_inv, alpha=self.params['c_1'] / 2)
                w /= np.linalg.norm(self.X @ w)
                self.u = self.X @ w
            elif self.method == 'john_constrained':
                w = constrained_regression(self.X, target, self.params['c_1'])
                self.u = self.X @ w
            elif self.method == 'mai':
                target = self.v
                w = self.lasso_solver(self.X, target, X_inv, alpha=self.params['c_1'])
                w /= np.linalg.norm(self.X @ w)
                self.u = self.X @ w

            # Update C
            if self.method == 'l2':
                u = self.X @ w
                c = self.ridge_solver(self.Y, u, Y_inv, alpha=self.params['c_2'])
                v = self.Y @ c
            elif self.method == 'witten':
                c = self.C.T @ w
                c, c_success = self.update(c, self.params['c_2'])
                if not c_success:
                    c = self.C.T @ w
                    c /= np.linalg.norm(c)
            elif self.method == 'parkhomenko':
                c = self.C.T @ w
                if np.linalg.norm(c) == 0:
                    c = self.C.T @ w
                c /= np.linalg.norm(c)
                c = self.soft_threshold(c, self.params['c_2'] / 2)
                if np.linalg.norm(c) == 0:
                    c = self.C.T @ w
                c /= np.linalg.norm(c)
            elif self.method == 'SAR':
                u = self.X @ w
                c = self.lasso_solver(self.Y, u, alpha=self.params['c_2'])
                # constraint
                c /= np.linalg.norm(c)
                v = self.Y @ c
            elif self.method == 'waaijenborg':
                u = self.X @ w
                c = self.elastic_solver(self.Y, u, alpha=self.params['c_2'], l1_ratio=self.params['l1_ratio_2'])
                # constraint
                c /= np.linalg.norm(Y @ c)
                v = self.Y @ c
            elif self.method == 'james':
                u = self.X @ w
                c = constrained_regression(self.Y, u, self.params['c_2'])
                v = self.Y @ c
            elif self.method == 'james_elastic':
                u = self.X @ w
                self.c_success, c = elastic_search(self.Y, u, 0.9, self.params['c_2'])
                v = self.Y @ c
            elif self.method == 'john':
                c = self.lasso_solver(self.Y, target, Y_inv, alpha=self.params['c_2'] / 2)
                c /= np.linalg.norm(self.Y @ c)
                self.v = self.Y @ c
                target = (self.v + self.u) / 2
            elif self.method == 'john_constrained':
                c = constrained_regression(self.Y, target, self.params['c_2'])
                self.v = self.Y @ c
                target = (self.v + self.u) / 2
            elif self.method == 'mai':
                target = self.u
                c = self.lasso_solver(self.Y, target, Y_inv, alpha=self.params['c_2'])
                c /= np.linalg.norm(self.Y @ c)
                self.v = self.Y @ c

            self.w = w
            self.c = c
            if not np.any(self.w):
                self.w_success = False
            if not np.any(self.c):
                self.c_success = False
            if not self.w_success or not self.c_success:
                return self

            # Check for convergence
            if _ > 0:
                if np.linalg.norm(self.w - w_old) < self.tol and np.linalg.norm(self.c - c_old) < self.tol:
                    break

            # Update trackers
            self.track_correlation[_] = np.corrcoef(self.X @ w, self.Y @ c)[0, 1]
            self.track_covariance[_] = np.cov(self.X @ w, self.Y @ c)[0, 1]
            self.track_distance[_] = np.linalg.norm(self.X @ w - self.Y @ c)
            self.track_norms_w[_] = np.linalg.norm(w, ord=1)
            self.track_norms_c[_] = np.linalg.norm(c, ord=1)
            # self.track_obj[_] = self.elastic_lyuponov(0.5, 0.5)
            self.track_obj[_] = self.lasso_lyuponov()
            self.alt_obj[_] = self.alt_lyuponov()

            w_old = w
            c_old = c
        return self

    def soft_threshold(self, x, delta):
        diff = abs(x) - delta
        diff[diff < 0] = 0
        out = np.sign(x) * diff
        return out

        # Find the first one outside the sphere
        first_outside_sphere = np.argmax(l2_norms >= 1)
        # Find the first one inside the sphere
        last_inside_sphere = first_outside_sphere - 1
        scaling_factor = (1 - l2_norms[last_inside_sphere]) / (
                l2_norms[first_outside_sphere] - l2_norms[last_inside_sphere])
        weights = weights[:, last_inside_sphere] + scaling_factor * (
                weights[:, first_outside_sphere] - weights[:, last_inside_sphere])
        return weights

    def bisec(self, K, c, x1, x2):
        # does a binary search between x1 and x2 (thresholds). Outputs weight vector.
        converge = False
        success = True
        tol = 1e-5
        while not converge and success:
            x = (x2 + x1) / 2
            out = self.soft_threshold(K, x)
            if np.linalg.norm(out, 2) > 0:
                if not self.method == 'james':
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

    def update(self, w, c):
        success = True
        delta = 0
        up = w
        # update the weights
        # Do the soft thresholding operation
        if not self.method == 'james':
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

    # @ignore_warnings(category=ConvergenceWarning)
    def lasso_solver(self, X, y, X_inv, alpha=0.1):
        if alpha == 0:
            if X_inv is not None:
                beta = np.dot(X_inv, y)
            else:
                clf = LinearRegression(fit_intercept=False)
                clf.fit(X, y)
                beta = clf.coef_
        else:
            lassoreg = Lasso(alpha=alpha, fit_intercept=False)
            lassoreg.fit(X, y)
            beta = lassoreg.coef_
        return beta

    # @ignore_warnings(category=ConvergenceWarning)
    def ridge_solver(self, X, y, X_inv=None, alpha=0):
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

    # @ignore_warnings(category=ConvergenceWarning)
    def elastic_solver(self, X, y, alpha=0.1, l1_ratio=0.5):
        clf = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        clf.fit(X, y)
        beta = clf.coef_
        if not np.any(beta):
            beta = np.ones(beta.shape)
        return beta

    def elastic_lyuponov(self, ratio_1, ratio_2):
        lyuponov = (self.X @ self.w).T @ (self.Y @ self.c) / (2 * self.X.shape[0]) + self.params[
            'c_1'] * ratio_1 * np.linalg.norm(
            self.w, ord=1) + self.params['c_2'] * ratio_2 * np.linalg.norm(self.c, ord=1) + 0.5 * self.params['c_1'] * (
                           1 - ratio_1) * self.w.T @ self.w + 0.5 * self.params['c_2'] * (
                           1 - ratio_2) * self.c.T @ self.c
        return lyuponov

    def lasso_lyuponov(self):
        lyuponov = 2 / (2 * self.X.shape[0]) - 2 * (self.X @ self.w).T @ (self.Y @ self.c) / (2 * self.X.shape[0]) + \
                   self.params[
                       'c_1'] * np.linalg.norm(
            self.w, ord=1) + self.params['c_2'] * np.linalg.norm(self.c, ord=1)
        return lyuponov

    def alt_lyuponov(self):
        target = ((self.X @ self.w) + (self.Y @ self.c)) / 2
        lyuponov = 2 * (2 / (2 * self.X.shape[0]) - 2 * (self.X @ self.w).T @ target / (2 * self.X.shape[0]) - 2 * (
                    self.Y @ self.c).T @ target / (2 * self.X.shape[0]) + 2 * target @ target / (2 * self.X.shape[0])) + \
                   self.params[
                       'c_1'] * np.linalg.norm(
            self.w, ord=1) + self.params['c_2'] * np.linalg.norm(self.c, ord=1)
        return lyuponov


def dist_spar_plot(track_distance, track_active_weights_w, track_active_weights_c, params, title, dist='correlation',
                   max_norm=100,max_obj=1):
    fig, axs = plt.subplots(1, track_active_weights_w.shape[0], sharey=True)
    axs[0].set_ylabel(dist)
    axs_2 = [a.twinx() for a in axs]
    axs_2[-1].set_ylabel('|w|_1')
    for p in range(track_active_weights_w.shape[0]):
        axs[p].plot(track_distance[p, :].T, color='k')
        axs_2[p].plot(track_active_weights_w[p, :].T, label='view 1 weights', linestyle=':')
        axs_2[p].plot(track_active_weights_c[p, :].T, label='view 2 weights', linestyle=':')
        # if dist == 'correlation':
        axs[p].set_ylim(bottom=0, top=max_obj)
        # else:
        #    axs[p].set_yscale('log')
        axs_2[p].set_ylim(bottom=0, top=max_norm)
        if p == track_active_weights_w.shape[0] - 1:
            axs[p].set_title('Normal ALS')
        else:
            axs[p].set_title('c={:.2E}'.format(params[p]))
        handles, labels = axs_2[p].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right')
    fig.suptitle(title)


def convergence_test(X, Y, max_iters=200, parameter_grid=None, method='l2'):
    # parameter grid needs to be kx2 where k is the number of pairs of parameters to try
    # X and Y are data (N x _ )
    # method chooses the kind of loop: 'l2', 'witten', 'parkhomenko', 'agoston', 'john', 'sar', 'waaijenborg'
    if parameter_grid is None:
        parameter_grid = [[0], [0]]

    length = parameter_grid.shape[0] + 1

    # Bunch of objects to store outputs
    track_distance = np.empty((length, max_iters))
    track_correlations = np.empty((length, max_iters))
    track_covariance = np.empty((length, max_iters))
    track_norms_w = np.zeros((length, max_iters))
    track_norms_c = np.zeros((length, max_iters))
    track_obj = np.zeros((length, max_iters))
    w = np.zeros((length, X.shape[1]))
    c = np.zeros((length, Y.shape[1]))
    # For each of the sets of parameters
    for l in range(length - 1):
        params = {'c_1': parameter_grid[l, 0], 'c_2': parameter_grid[l, 1], 'l1_ratio_1': 0.5, 'l1_ratio_2': 0.5}
        # This uses my ALS_inner_loop class
        ALS = ALS_inner_loop(X, Y, method=method, params=params, max_iter=max_iters)
        # Get the statistics
        track_distance[l, :] = ALS.track_distance
        track_correlations[l, :] = ALS.track_correlation
        track_covariance[l, :] = ALS.track_covariance
        track_norms_w[l, :] = ALS.track_norms_w
        track_norms_c[l, :] = ALS.track_norms_c
        # Useful for the John one
        track_obj[l, :] = ALS.track_obj
        w[l, :] = ALS.w
        c[l, :] = ALS.c

    # Run an ALS CCA for comparison
    ALS = ALS_inner_loop(X, Y, method='l2', params=None, max_iter=max_iters)
    track_distance[-1, :] = ALS.track_distance
    track_correlations[-1, :] = ALS.track_correlation
    track_covariance[-1, :] = ALS.track_covariance
    track_norms_w[-1, :] = ALS.track_norms_w
    track_norms_c[-1, :] = ALS.track_norms_c
    #track_obj /= np.nanmax(track_obj)
    return track_distance, track_correlations, track_covariance, track_norms_w, track_norms_c, track_obj


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

    if current_val < 0:
        if previous_val < 0:
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


X = np.random.rand(100, 50)
X -= X.mean(axis=0)
X /= np.linalg.norm(X, axis=0)
Y = np.random.rand(100, 50)
Y -= Y.mean(axis=0)
Y /= np.linalg.norm(Y, axis=0)

X,Y=generate_candola(100, 1, 500, 500, 0.2, 0.2, sparse=True)


number_of_parameters_to_try = 5

parameter_grid = np.ones((number_of_parameters_to_try, 2)) * 0
parameter_grid[:, 0] = np.linspace(1e-4, 1e-3, number_of_parameters_to_try)
parameter_grid[:, 1] = np.linspace(1e-4, 1e-3, number_of_parameters_to_try)

james = convergence_test(X, Y, max_iters=200, parameter_grid=parameter_grid, method='mai')
john = convergence_test(X, Y, max_iters=200, parameter_grid=parameter_grid, method='john')

max_norm = max(np.nanmax(james[3]), np.nanmax(james[4]),
               np.nanmax(john[3]), np.nanmax(john[4]))
max(np.nanmax(james[3]), np.nanmax(james[4]),
               np.nanmax(john[3]), np.nanmax(john[4]))
max_obj = max(np.nanmax(james[5]), np.nanmax(john[5]))

dist_spar_plot(james[5], james[3], james[4], parameter_grid[:, 0], 'Without Averaging', dist='Objective Function',
               max_norm=max_norm,max_obj=max_obj)

dist_spar_plot(john[5], john[3], john[4], parameter_grid[:, 0], 'With Averaging',
               dist='Objective Function',
               max_norm=max_norm,max_obj=max_obj)

