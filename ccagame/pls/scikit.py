import numpy as np
from sklearn.cross_decomposition import PLSSVD


def calc_sklearn(X, Y, k):
    pls = PLSSVD(n_components=k, scale=False).fit(np.array(X), np.array(Y))
    plsx, plsy = pls.transform(X, Y)
    pls_cov = np.diag(np.corrcoef(plsx, plsy, rowvar=False)[k:, :k])
    return pls_cov, pls.x_weights_, pls.y_weights_
