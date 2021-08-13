from sklearn.cross_decomposition import CCA
import numpy as np

def calc_sklearn(X, Y, n):
    cca = CCA(n_components=n, scale=False).fit(np.array(X), np.array(Y))
    ccax, ccay = cca.transform(X, Y)
    cca_corr = np.diag(np.corrcoef(ccax, ccay, rowvar=False)[n:, :n])
    return cca_corr, cca.x_weights_, cca.y_weights_
