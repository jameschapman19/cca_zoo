from time import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import lars_path


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


def constrained_regression(X, y, p, search='bin', tol=1e-7):
    if p == 0:
        coef = SGDRegressor(fit_intercept=False).fit(X, y).coef_
    if search == 'bin':
        converged = False
        min_ = 0
        max_ = 1
        current = 1
        previous = 1
        previous_val = 1e+10
        i = 0
        while not converged:
            i += 1
            coef = Lasso(alpha=current, selection='cyclic').fit(X, y).coef_
            if np.linalg.norm(X @ coef) > 0:
                current_val = p - np.linalg.norm(coef / np.linalg.norm(X @ coef), ord=1)
            else:
                current_val = p
            current, previous, min_, max_ = bin_search(current, previous, current_val, previous_val, min_, max_)
            previous_val = current_val
            if np.abs(current_val) < tol:
                converged = True
            elif np.abs(max_ - min_) < 1e-15 or i == 50:
                converged = True
                coef = Lasso(alpha=min_, selection='cyclic').fit(X, y).coef_
    else:
        converged = True
        alphas, _, coefs = lars_path(X, y)
        coefs_l1 = np.linalg.norm(coefs, ord=1, axis=0)
        projs_l2 = np.linalg.norm(X @ coefs, ord=2, axis=0)
        projs_l2[projs_l2 == 0] = 1e-20
        l1_normalized_norms = coefs_l1 / projs_l2
        l1_normalized_norms[np.isnan(l1_normalized_norms)] = 0
        alpha_max = np.argmax(l1_normalized_norms > p)
        alpha_min = alpha_max - 1
        if alpha_max == 0:  # -1:
            coef = SGDRegressor(fit_intercept=False).fit(X, y).coef_
        elif alpha_min == 0:
            converged = False
            print('Warning: increase p')
        else:
            new_vec_hat = X @ (coefs[:, alpha_max] - coefs[:, alpha_min])
            old_vec = X @ coefs[:, alpha_min]
            m = coefs_l1[alpha_max] - coefs_l1[alpha_min]
            c = coefs_l1[alpha_min]
            A = m ** 2 - p ** 2 * new_vec_hat.T @ new_vec_hat
            B = 2 * m * c - 2 * p ** 2 * new_vec_hat.T @ old_vec
            C = c ** 2 - p ** 2 * old_vec.T @ old_vec
            f_pos = (-B + np.sqrt(B ** 2 - 4 * A * C)) / (2 * A)
            coef = coefs[:, alpha_min] + f_pos * (coefs[:, alpha_max] - coefs[:, alpha_min])
    coef = coef / np.linalg.norm(X @ coef)
    return coef, converged


def speed_run(runs=1, feat=100, p=0.1, tol=1e-10, subplot=None, axs=None):
    times = np.zeros((runs, 2))
    for run in range(runs):
        X = np.random.rand(1000, feat)
        y = np.random.rand(1000, 1)
        t0 = time()
        constrained_regression(X, y, p, search='bin')
        times[run, 0] = time() - t0
    for run in range(runs):
        X = np.random.rand(1000, feat)
        y = np.random.rand(1000)
        t0 = time()
        constrained_regression(X, y, p, search='path')
        times[run, 1] = time() - t0
    if axs is not None:
        axs[subplot].hist(times[:, 0], label='binary search', alpha=0.5, density=True)
        axs[subplot].hist(times[:, 1], label='lasso path', alpha=0.5, density=True)
        axs[subplot].set_xlabel('Time (s)')
        if subplot[1] == 0:
            axs[subplot].set_ylabel(str(feat) + " features")
        if subplot[0] == 0:
            axs[subplot].set_title("tol: " + str(tol))
    print(times[:, 0].mean())
    print(times[:, 1].mean())


def tol_dim_fig(n=2, runs=1):
    feat_list = np.linspace(100, 2000, n, dtype=int)
    tol_list = [10 ** -(k + 5) for k in range(n)]
    plot = 0
    fig, axs = plt.subplots(ncols=n, nrows=n, sharex='row', figsize=(10, 10))
    for f in range(len(feat_list)):
        for t in range(len(tol_list)):
            speed_run(runs=runs, feat=feat_list[f], tol=tol_list[t], subplot=(f, t), axs=axs)
    plt.legend()
    plt.suptitle('Speed to converge to within tolerance for different feature size')
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)


X = np.random.rand(1000, 900)
y = np.random.rand(1000)
X /= X.std(axis=0)
y /= y.std(axis=0)
p = 0.02
coef = constrained_regression(X, y, p, search='path', tol=1e-7)

# speed_run(runs=1, feat=100, tol=1e-3)

tol_dim_fig(n=5, runs=50)
plt.savefig('Speedtest')
plt.show()
