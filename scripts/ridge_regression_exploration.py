import numpy as np

from sklearn.linear_model import LinearRegression, ElasticNet


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


def constrained_elastic(X, y, alpha=0.1, l1_ratio=0.5, init=0):
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
            # coef = Lasso(alpha=min_, selection='cyclic', max_iter=10000).fit(X, y).coef_
            coef = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False).fit(np.sqrt(current + 1) * X,
                                                                                       y / np.sqrt(current + 1)).coef_
    return coef, current


X = np.random.rand(100, 50)
X -= X.mean(axis=0)
X /= X.std(axis=0)
y = np.random.rand(100, 1)
y -= y.mean(axis=0)
y /= y.std(axis=0)

alphas = np.arange(-1, 1, 0.1)
alphas = 10 ** alphas

scores = np.zeros((X.shape[0], len(alphas)))
coefs = np.zeros((X.shape[1], len(alphas)))
coefs_constrained = np.zeros((X.shape[1], len(alphas)))

l1_ratio = 1

for i, alpha in enumerate(alphas):
    coefs[:, i] = ElasticNet(alpha=alpha, l1_ratio=l1_ratio).fit(X, y).coef_
    coefs_constrained[:, i] = constrained_elastic(X, y, alpha=alpha, l1_ratio=l1_ratio)[0]

scores = X @ coefs

lengths = np.linalg.norm(scores, axis=0)

norm_scores = scores / lengths

norm_lengths = np.linalg.norm(norm_scores, axis=0)

constrained_scores = X @ coefs_constrained

constrained_lengths = np.linalg.norm(constrained_scores, axis=0)

print('here')
