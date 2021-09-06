# Importing necessary libraries
import jax.numpy as jnp
import numpy as np

from ccagame.pca._pca import _PCA


#object form
class Numpy(_PCA):
    def __init__(self, n_components=2, *, scale=True, copy=True, lr: float = 1, epochs: int = 100,
                 random_state: int = 0, verbose=False):
        super().__init__(n_components, scale=scale, copy=copy)
        self.lr = lr
        self.epochs = epochs
        self.random_state = random_state
        self.verbose = verbose

    def _fit(self, X):
        eigvals, eigvecs = jnp.linalg.eigh(X.T @ X)
        idx = np.argsort(eigvals, axis=0)[::-1][:self.n_components]
        eigvecs = eigvecs[:, idx]
        return eigvecs

# function form
def calc_numpy(X, k):
    eigvals, eigvecs = jnp.linalg.eigh(X.T @ X)
    idx = np.argsort(eigvals, axis=0)[::-1][:k]
    eigvecs = eigvecs[:, idx]
    eigvals = eigvals[idx]
    return eigvals, eigvecs