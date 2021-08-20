import numpy as np
from sklearn.decomposition import PCA


def calc_sklearn(X, k):
    pca = PCA(n_components=k).fit(np.array(X))
    return pca.singular_values_[:k] ** 2, pca.components_[:k].T
