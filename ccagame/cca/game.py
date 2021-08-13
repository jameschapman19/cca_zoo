# Importing necessary libraries
from functools import partial

import jax.numpy as jnp
from jax import jit, grad

from .utils import initialize


# Define utlity function, we will take grad of this in the
# update step, v is the current eigenvector being calculated
# X is the design matrix and V1 holds the previously computed eigenvectors
@partial(jit, static_argnums=(5))
def model(u, v, X, Y, V1, k):
    C_xy = jnp.dot(jnp.transpose(X), Y)
    C_xx = jnp.dot(jnp.transpose(X), X)
    C_yy = jnp.dot(jnp.transpose(Y), Y)
    # rewards = jnp.dot(jnp.transpose(u), jnp.dot(C_xy, v)) / (
    #        jnp.sqrt(jnp.dot(jnp.transpose(u), jnp.dot(C_xx, u))) * jnp.sqrt(
    #    jnp.dot(jnp.transpose(v), jnp.dot(C_yy, v))))
    rewards = jnp.dot(jnp.transpose(u), jnp.dot(C_xy, v)) ** 2 / (
            jnp.dot(jnp.transpose(u), jnp.dot(C_xx, u)) * jnp.dot(jnp.transpose(v), jnp.dot(C_yy, v)))
    penalties = 0
    for j in range(k):
        penalties = penalties + jnp.dot(jnp.transpose(u), jnp.dot(C_xy, V1[:, j].reshape(-1, 1))) ** 2 / (jnp.dot(
            jnp.transpose(V1[:, j].reshape(-1, 1)), jnp.dot(C_yy, V1[:, j].reshape(-1, 1))) * jnp.dot(
            jnp.transpose(u), jnp.dot(C_xx, u)))
    return jnp.sum(rewards - penalties)


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
    U1, V1 = initialize(X, Y, n, initialization, random_state)
    if simultaneous:
        for i in range(iterations):
            for k in range(n):
                u, v = update(U1[:, k], V1[:, k], X, Y, U1, V1, k, lr=lr, riemannian_projection=riemannian_projection)
                U1 = U1.at[:, k].set(u)
                V1 = V1.at[:, k].set(v)
            print(f'iteration {i}: {calc_eigengame_eigenvalues(X, Y, U1, V1)}')
    else:
        for k in range(n):
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
