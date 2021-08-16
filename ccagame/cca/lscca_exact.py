# Importing necessary libraries
import jax.numpy as jnp
from jax import jit
from .utils import initialize, calc_eigenvalues


# Update rule to be used for calculating eigenvectors
# For first eigenvector use riemannian_projection = False (update rule given in the paper doesn't work without the penalty term)
# For all others, use riemannian_projection = True to be aligned with the paper
# But using riemannian_projection = False also works and in the tests that I did it converges much faster than including the
# Riemannian Projection
@jit
def update(X, Y, Hx, Hy, U1, V1):
    vhat = jnp.dot(Hy, jnp.dot(X, U1))
    uhat = jnp.dot(Hx, jnp.dot(Y, V1))
    return jnp.linalg.qr(uhat)[0], jnp.linalg.qr(vhat)[0]


# Run the update step iteratively across all eigenvectors
def calc_lscca_exact(X, Y, n, iterations=100, initialization='uniform',
                     random_state=0):
    U1, V1 = initialize(X, Y, n, initialization, random_state)
    Hx = jnp.dot(jnp.linalg.inv(jnp.dot(jnp.transpose(X), X)), jnp.transpose(X))
    Hy = jnp.dot(jnp.linalg.inv(jnp.dot(jnp.transpose(Y), Y)), jnp.transpose(Y))
    for i in range(iterations):
        U1, V1 = update(X, Y, Hx, Hy, U1, V1)
        print(f'iteration {i}: {calc_eigenvalues(X, Y, U1, V1)}')
    return calc_eigenvalues(X, Y, U1, V1), U1, V1
