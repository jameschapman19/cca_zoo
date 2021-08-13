# Importing necessary libraries
import jax.numpy as jnp
from jax import grad
from jax import random


# Update rule to be used for calculating eigenvectors
# For first eigenvector use riemannian_projection = False (update rule given in the paper doesn't work without the penalty term)
# For all others, use riemannian_projection = True to be aligned with the paper
# But using riemannian_projection = False also works and in the tests that I did it converges much faster than including the
# Riemannian Projection
def update(u, v, X, Y, U1, V1, k, lr=1e-1, riemannian_projection=False):
    du = grad(model)(u, v, X, Y, V1, k)
    dv = grad(model)(v, u, Y, X, U1, k)
    if riemannian_projection:
        dur = du - (jnp.dot(du.T, u)) * u
        uhat = u + lr * dur
        dvr = dv - (jnp.dot(dv.T, v)) * v
        vhat = v + lr * dvr
    else:
        uhat = u + lr * du
        vhat = v + lr * dv
    return uhat / jnp.linalg.norm(uhat), vhat / jnp.linalg.norm(vhat)


# Run the update step iteratively across all eigenvectors
# Run the update step iteratively across all eigenvectors
def calc_eigengame(X, Y, n, lr=1e-1, iterations=100, riemannian_projection=False, initialization='random',
                   random_state=0, simultaneous=False):
    if initialization == 'svd':
        U1, _, V1 = jnp.linalg.svd(X.T @ Y)
        U1 = U1[:, :n]
        V1 = V1[:, :n]
    elif initialization == 'cca':
        _, U1, V1 = calc_sklearn(X, Y, n)
        U1 = jnp.array(U1)
        V1 = jnp.array(V1)
    elif initialization == 'random':
        key = random.PRNGKey(random_state)
        key, subkey = random.split(key)
        U1 = random.normal(key, (X.shape[1], n))
        V1 = random.normal(subkey, (Y.shape[1], n))
        U1 = U1 / jnp.linalg.norm(U1, axis=0)
        V1 = V1 / jnp.linalg.norm(V1, axis=0)
    else:
        print(f'Initialization "{initialization}" not implemented')
        return
    for i in range(iterations):
        u, v = update(U1[:, k], V1[:, k], X, Y, U1, V1, k, lr=lr, riemannian_projection=riemannian_projection)
        U1 = U1.at[:, k].set(u)
        V1 = V1.at[:, k].set(v)
        print(f'iteration {i}: {calc_eigengame_eigenvalues(X, Y, U1, V1)}')
    return calc_eigengame_eigenvalues(X, Y, U1, V1), U1, V1


# Calculate eigenvalues once the eigenvectors have been computed
def calc_eigengame_eigenvalues(X, Y, U1, V1):
    C_xy = jnp.dot(jnp.transpose(X), Y)
    C_xx = jnp.dot(jnp.transpose(X), X)
    C_yy = jnp.dot(jnp.transpose(Y), Y)
    n = jnp.size(V1, axis=1)
    eigvals = jnp.zeros((1, n))
    for k in range(n):
        eigvals = eigvals.at[:, k].set(jnp.dot(U1[:, k], jnp.dot(C_xy, V1[:, k].reshape(-1, 1))) / (
                jnp.sqrt(jnp.dot(U1[:, k], jnp.dot(C_xx, U1[:, k].reshape(-1, 1)))) * jnp.sqrt(
            jnp.dot(V1[:, k], jnp.dot(C_yy,
                                      V1[:, k].reshape(
                                          -1, 1))))))
    return eigvals
