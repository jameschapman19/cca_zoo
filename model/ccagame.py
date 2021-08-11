# Importing necessary libraries
import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp
from jax import random
from sklearn.cross_decomposition import CCA
from jax import jit, grad
from functools import partial


# Calculate the eigenvalues of covariance matrix of X using Numpy for comparison
def calc_numpy(X, Y, n, r=0):
    dof = X.shape[0] - 1
    C = jnp.hstack((X, Y))
    C = C.T @ C / dof
    # Get the block covariance matrix placing Xi^TX_i on the diagonal
    D = jsp.linalg.block_diag(
        *[m.T @ m + r * jnp.eye(m.shape[1]) for i, m in enumerate([X, Y])]) / dof

    C = C - jsp.linalg.block_diag(*[view.T @ view / dof for view in [X, Y]]) + D

    R = jnp.linalg.inv(jnp.linalg.cholesky(D))

    # In MCCA our eigenvalue problem Cv = lambda Dv
    C_whitened = R @ C @ R.T

    eigvals, eigvecs = jnp.linalg.eigh(C_whitened)
    idx = np.argsort(eigvals, axis=0)[::-1][:n]
    eigvecs = eigvecs[:, idx]
    return eigvals, eigvecs[:X.shape[1]], eigvecs[X.shape[1]:]

def calc_sklearn(X, Y, n):
    cca = CCA(n_components=n, scale=False).fit(np.array(X), np.array(Y))
    ccax, ccay = cca.transform(X, Y)
    cca_corr = np.diag(np.corrcoef(ccax, ccay, rowvar=False)[p:n+p, :n])
    return cca_corr, cca.x_weights_, cca.y_weights_

# Define utlity function, we will take grad of this in the
# update step, v is the current eigenvector being calculated
# X is the design matrix and V1 holds the previously computed eigenvectors
# @partial(jit, static_argnums=(5))
def model(u, v, X, Y, U1, k):
    C_xy = jnp.dot(jnp.transpose(X), Y)
    C_xx = jnp.dot(jnp.transpose(X), X)
    C_yy = jnp.dot(jnp.transpose(Y), Y)
    rewards = jnp.dot(jnp.transpose(u), jnp.dot(C_xy, v)) / (
            jnp.sqrt(jnp.dot(jnp.transpose(u), jnp.dot(C_xx, u))) * jnp.sqrt(
        jnp.dot(jnp.transpose(v), jnp.dot(C_yy, v))))
    penalties = 0
    for j in range(k):
        penalties = penalties + jnp.dot(jnp.transpose(u), jnp.dot(C_xx, U1[:, j].reshape(-1, 1))) ** 2 / (jnp.dot(
            jnp.transpose(U1[:, j].reshape(-1, 1)), jnp.dot(C_xx, U1[:, j].reshape(-1, 1))) * jnp.dot(
            jnp.transpose(u), jnp.dot(C_xx, u)))
    return jnp.sum(rewards - penalties)


# Update rule to be used for calculating eigenvectors
# For first eigenvector use riemannian_projection = False (update rule given in the paper doesn't work without the penalty term)
# For all others, use riemannian_projection = True to be aligned with the paper
# But using riemannian_projection = False also works and in the tests that I did it converges much faster than including the
# Riemannian Projection
def update(u, v, X, Y, U1, V1, k, lr=1e-1, riemannian_projection=False):
    du = grad(model)(u, v, X, Y, U1, k)
    dv = grad(model)(v, u, Y, X, V1, k)
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
                                random_state=0):
    if initialization == 'svd':
        U1, _, V1 = jnp.linalg.svd(X.T @ Y)
        U1 = U1[:, :n]
        V1 = V1[:, :n]
    elif initialization == 'cca':
        _,U1,V1=calc_sklearn(X,Y,n)
        U1=jnp.array(U1)
        V1=jnp.array(V1)
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
        for k in range(n):
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
        eigvals=eigvals.at[:, k].set(jnp.dot(U1[:, k], jnp.dot(C_xy, V1[:, k].reshape(-1, 1))) / (
                jnp.sqrt(jnp.dot(U1[:, k], jnp.dot(C_xx, U1[:, k].reshape(-1, 1)))) * jnp.sqrt(
            jnp.dot(V1[:, k], jnp.dot(C_yy,
                                      V1[:, k].reshape(
                                          -1, 1))))))
    return eigvals

