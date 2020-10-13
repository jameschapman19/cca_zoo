import matplotlib
import numpy as np
from scipy.linalg import pinv2
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class ALS_inner_loop:
    # an alternating least squares inner loop
    def __init__(self, X, Y, confounds=None, max_iter=100, tol=np.finfo(np.float32).eps, initialization='random',
                 params=None,
                 method='l2'):
        self.initialization = initialization
        self.X = X
        self.Y = Y
        self.C = X.T @ Y
        self.max_iter = max_iter
        self.tol = tol
        self.params = params
        self.confounds = confounds
        if params is None:
            self.params = {'c_1': 0, 'c_2': 0}
        self.method = method
        self.iterate()

    def iterate(self):
        # Any initialization required

        # Weight vectors for y (normalized to 1)
        c = np.random.rand(self.X.shape[1])
        c /= np.linalg.norm(c)

        # Score/Projection vector for Y (normalized to 1)
        v = np.random.rand(self.X.shape[0])
        v = v / np.linalg.norm(v)

        u = None

        self.aux = v
        f = c

        # Pre calculate inverses. Used for unregularized CCA_archive
        X_inv = pinv2(self.X)
        Y_inv = pinv2(self.Y)
        if self.confounds is not None:
            confounds_inv = pinv2(self.confounds)

        alpha_w = self.X.shape[1]
        alpha_c = self.Y.shape[1]

        # Object to store statistics for convergence analysis
        self.track_correlation = np.zeros(self.max_iter)
        self.track_correlation[:] = np.nan
        self.track_covariance = np.zeros(self.max_iter)
        self.track_covariance[:] = np.nan
        self.track_distance = np.zeros(self.max_iter)
        self.track_distance[:] = np.nan
        # Useful for the John one
        self.track_alpha_w = np.zeros(self.max_iter)
        self.track_alpha_w[:] = np.nan
        self.track_alpha_c = np.zeros(self.max_iter)
        self.track_alpha_c[:] = np.nan
        # Useful for the John one
        self.track_active_weights_w = np.zeros(self.max_iter)
        self.track_active_weights_w[:] = np.nan
        self.track_active_weights_c = np.zeros(self.max_iter)
        self.track_active_weights_c[:] = np.nan

        self.track_confound_correlation_w = np.zeros(self.max_iter)
        self.track_confound_correlation_c = np.zeros(self.max_iter)
        self.track_confound_correlation_aux = np.zeros(self.max_iter)

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
                w /= np.linalg.norm(self.X @ w)
            elif self.method == 'dcf':
                # Use ridge solver or pseudo-inverse
                w = self.ridge_solver(self.X, self.aux, X_inv, alpha=self.params['c_1'])
                w /= np.linalg.norm(self.X @ w)

            # TODO could we group the w and c update to prevent the duplication?

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
                c /= np.linalg.norm(self.self.Y @ c)
                v = self.Y @ c
            elif self.method == 'dcf':
                u = self.X @ w
                c = self.ridge_solver(self.Y, self.aux, Y_inv, alpha=self.params['c_2'])
                c /= np.linalg.norm(self.Y @ c)
                v = self.Y @ c
                self.f = self.ridge_solver(self.confounds, self.aux, confounds_inv)
                self.f /= np.linalg.norm(self.confounds @ self.f)
                self.aux = u + v
                self.aux = self.aux - self.aux @ (self.confounds @ self.f) * self.confounds @ self.f
                self.aux /= np.linalg.norm(self.aux)

            if self.method is not 'dcf':
                self.aux = u + v
                self.aux /= np.linalg.norm(self.aux)
                self.f = self.ridge_solver(self.confounds, self.aux, confounds_inv)

            # Check for convergence
            if _ > 0:
                if np.linalg.norm(w - w_old) < self.tol and np.linalg.norm(c - c_old) < self.tol:
                    break

            w_old = w
            c_old = c

            # Update trackers
            self.track_correlation[_] = np.corrcoef(self.X @ w, self.Y @ c)[0, 1]
            self.track_covariance[_] = np.cov(self.X @ w, self.Y @ c)[0, 1]
            self.track_distance[_] = np.linalg.norm(self.X @ w - self.Y @ c)
            self.track_confound_correlation_w[_] = np.corrcoef(self.confounds @ self.f, self.X @ w)[0, 1]
            self.track_confound_correlation_c[_] = np.corrcoef(self.confounds @ self.f, self.Y @ c)[0, 1]
            self.track_confound_correlation_aux[_] = np.corrcoef(self.confounds @ self.f, self.aux)[0, 1]
            self.track_active_weights_w[_] = np.count_nonzero(w)
            self.track_active_weights_c[_] = np.count_nonzero(c)
            self.track_alpha_w[_] = alpha_w
            self.track_alpha_c[_] = alpha_c
        self.w = w
        self.c = c
        return self

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

    def update(self, w, c):
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

    # @ignore_warnings(category=ConvergenceWarning)
    def lasso_solver(self, X, y, alpha=0.1):
        lassoreg = Lasso(alpha=alpha, fit_intercept=False)
        lassoreg.fit(X, y)
        beta = lassoreg.coef_
        if not np.any(beta):
            beta = np.ones(beta.shape)
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


X = np.random.rand(500, 20)
X -= X.mean(axis=0)
Y = np.random.rand(500, 20)
Y -= Y.mean(axis=0)
C = np.random.rand(500, 20)
C -= C.mean(axis=0)
C[:,10:]=C[:,:10]**2

a = ALS_inner_loop(X, Y, confounds=C, method='dcf', max_iter=500)

X_aug = X - C @ pinv2(C.T @ C) @ C.T @ X

Y_aug = Y - C @ pinv2(C.T @ C) @ C.T @ Y

b = ALS_inner_loop(X_aug, Y_aug, confounds=C, method='l2', max_iter=500)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(X@a.w,Y@a.c,C@a.f)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(X@b.w,Y@b.c,C@b.f)
plt.show()
print('hello')
