import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import ElasticNet, enet_path, LinearRegression

plt.style.use('ggplot')
X = np.random.rand(1000, 20).astype(np.float32)
X -= X.mean(axis=0)
X /= np.linalg.norm(X, axis=0)

y = np.random.rand(1000).astype(np.float32)
y /= np.linalg.norm(y)


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


def elastic_search(X, y, l1_ratio, alpha, n_tries=100):
    # Grid search for lam_1 and lam_2. Hmm.
    success=False
    tries=0
    tol=1e-5
    mu = -1+tol
    mu_old = mu
    length_old=0
    mu_min=-1
    mu_max=0
    coef = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False).fit(np.sqrt(1 + mu) * X,
                                                                               y / np.sqrt(1 + mu)).coef_
    length_old = np.linalg.norm(X @ coef)
    while not success and tries < n_tries:
        tries += 1
        coef = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False).fit(np.sqrt(1+mu)*X, y/np.sqrt(1+mu)).coef_

        length = np.linalg.norm(X @ coef)

        if np.abs(length-1)<tol:
            success=True
        if not success:
            mu, mu_old, mu_min, mu_max = bin_search(mu, mu_old,1-length,1-length_old, mu_min,mu_max)
            length_old=length
    return coef, success

def lagrange_elastic(X, y, l1_ratio_star, alpha_star, n_tries=100):
    # Grid search for lam_1 and lam_2. Hmm.
    l1_ratio = 1
    l1_ratio_old = l1_ratio
    l1_ratio_min = 0.01
    tol=1e-5
    l1_ratio_max = 1
    tol = 0.01
    alpha_high = 0
    alpha_high_old = alpha_high
    success = False
    tries = 0
    lower=0
    upper=0
    alpha_low=0
    while not success and tries < n_tries:
        tries += 1
        alphas_enet, coefs_enet, _ = enet_path(
            X, y, l1_ratio=l1_ratio, fit_intercept=False)

        mu = np.linalg.norm(X @ coefs_enet, axis=0) - 1

        l1_ratios_eff = l1_ratio / (1 + mu - mu * l1_ratio)
        alphas_eff = alphas_enet * (1 + mu - mu * l1_ratio)
        if l1_ratios_eff.min() < l1_ratio_star and l1_ratios_eff.max() > l1_ratio_star:
            lower = np.where(l1_ratios_eff < l1_ratio_star)[0][0]
            upper = np.where(l1_ratios_eff > l1_ratio_star)[0][-1]
            alpha_low = alphas_eff[lower]
            alpha_high = alphas_eff[upper]
            if alpha_low < alpha_star and alpha_high > alpha_star:
                success = True

        if not success:
            l1_ratio, l1_ratio_old, l1_ratio_min, l1_ratio_max = bin_search(l1_ratio, l1_ratio_old,
                                                                            alpha_star - alpha_high,
                                                                            alpha_star - alpha_high_old, l1_ratio_min,
                                                                            l1_ratio_max)
            alpha_high_old = alpha_high
            if l1_ratio-0.01<tol:
                break

    coefs=coefs_enet/np.linalg.norm(X@coefs_enet[:,upper])

    return success, coefs, l1_ratios_eff[upper],alpha_high


def constrained_elastic_regression(X, y, p, ratio=0.1):
    if p == 0:
        coef_star = LinearRegression(fit_intercept=False).fit(X, y).coef_
    else:
        alphas, coefs, _ = enet_path(
            X, y, l1_ratio=ratio, fit_intercept=False)
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

def lagrange_elastic_archive(X, y, l1_ratio_star, alpha_star):
    # Grid search for lam_1 and lam_2. Hmm.
    l1_ratio = 0.1
    l1_ratio = l1_ratio
    tol = 1e-5
    success = False

    alphas_enet, coefs_enet, _ = enet_path(
        X, y, l1_ratio=l1_ratio, fit_intercept=False)

    mu = np.linalg.norm(X @ coefs_enet, axis=0) - 1

    l1_ratios_eff = l1_ratio / (1 + mu - mu * l1_ratio)
    alphas_eff = alphas_enet * (1 + mu - mu * l1_ratio)

    if l1_ratios_eff.min() < l1_ratio_star and l1_ratios_eff.max() > l1_ratio_star:
        upper_bound = np.where(l1_ratios_eff < l1_ratio_star)[0][0]
        lower_bound = np.where(l1_ratios_eff > l1_ratio_star)[0][-1]
        grid[l, 0] = alphas_eff[lower_bound]
        grid[l, 1] = alphas_eff[upper_bound]

    return grid


coef,success=elastic_search(X, y, 0.5, 0.00001, n_tries=100)

print('here')
