# Importing necessary libraries
import jax.numpy as jnp

from .utils import initialize


# Update rule to be used for calculating eigenvectors
# For first eigenvector use riemannian_projection = False (update rule given in the paper doesn't work without the penalty term)
# For all others, use riemannian_projection = True to be aligned with the paper
# But using riemannian_projection = False also works and in the tests that I did it converges much faster than including the
# Riemannian Projection
def update(u, v, X, Y, U1, V1, k, lr=1e-1, riemannian_projection=False):
    return uhat / jnp.linalg.norm(uhat), vhat / jnp.linalg.norm(vhat)


# Run the update step iteratively across all eigenvectors
# Run the update step iteratively across all eigenvectors
def calc_eigengame(X, Y, n, lr=1e-1, iterations=100, riemannian_projection=False, initialization='random',
                   random_state=0):
    U1, V1 = initialize(X, Y, n, initialization, random_state)
    for i in range(iterations):
        U1, V1 = update(U1, V1, X, Y, U1, V1, lr=lr, riemannian_projection=riemannian_projection)
        print(f'iteration {i}: {calc_eigenvalues(X, Y, U1, V1)}')
    return calc_eigengame_eigenvalues(X, Y, U1, V1), U1, V1
