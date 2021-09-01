"""
Efficient Globally Convergent Stochastic
Optimization for Canonical Correlation Analysis
https://proceedings.neurips.cc/paper/2016/file/42998cf32d552343bc8e460416382dca-Paper.pdf
"""
# Importing necessary libraries
import jax.numpy as jnp
from jax import jit

from .utils import initialize, TCC
from . import _CCA

# Update rule to be used for calculating eigenvectors
@jit
def update(X, Y, U, V):
    vhat = jnp.linalg.lstsq(Y, X @ U)[0]
    uhat = jnp.linalg.lstsq(X, Y @ V)[0]
    return jnp.linalg.qr(uhat)[0], jnp.linalg.qr(vhat)[0]


# Run the update step iteratively across all eigenvectors
def calc_lscca(X, Y, k, iterations=100, initialization='uniform',
               random_state=0):
    U, V = initialize(X, Y, k, initialization, random_state)
    for i in range(iterations):
        U, V = update(X, Y, U, V)
    return TCC(X, Y, U, V), U, V

class AlternatingLeastSquares(_CCA):

    def __init__(self, n_components=4, *, scale=True, copy=True, epochs: int = 100,
                 random_state: int = 0, batch_size: int = 128, verbose=False, lr=1):
        super().__init__(n_components, scale=scale, copy=copy)
        self.epochs = epochs
        self.random_state = random_state
        self.batch_size = batch_size
        self.verbose = verbose
        self.lr=lr


    def _fit(self, X, Y):
        U, V = initialize(X, Y, self.n_components, 'random', self.random_state)
        for i in range(self.epochs):
            U, V = update(X, Y, U, V)
            print(f'iteration {i}: {TCC(X, Y, U, V)}')
        return U, V