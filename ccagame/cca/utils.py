import jax.numpy as jnp
import jax.scipy as jsp
from jax import random


def TCC(X, Y, Wx, Wy):
    dof = X.shape[0] - 1
    X_hat = X @ Wx
    Y_hat = Y @ Wy
    C = jnp.hstack((X_hat, Y_hat))
    C = C.T @ C / dof
    # Get the block covariance matrix placing Xi^TX_i on the diagonal
    D = jsp.linalg.block_diag(
        *[m.T @ m for i, m in enumerate([X_hat, Y_hat])]) / dof
    C = C - jsp.linalg.block_diag(*[view.T @ view / dof for view in [X_hat, Y_hat]]) + D
    R = jnp.linalg.inv(jnp.linalg.cholesky(D))
    # In MCCA our eigenvalue problem Cv = lambda Dv
    C_whitened = R @ C @ R.T
    eigvals = jnp.linalg.eigvalsh(C_whitened)[::-1][:Wx.shape[1]]-1
    return eigvals.real.sum()


def gram_schmidt_matrix(W, M):
    for k in range(W.shape[1]):
        C = jnp.zeros((W.shape[0], k))
        for j in range(k):
            C = C.at[:, j].set(jnp.dot(W[:, j], jnp.dot(jnp.transpose(W[:, k]), jnp.dot(M, W[:, j]))))
        W = W.at[:, k].set(W[:, k] - jnp.sum(C, axis=1))
        W = W.at[:, k].set(W[:, k] / jnp.sqrt(jnp.dot(jnp.transpose(W[:, k]), jnp.dot(M, W[:, k]))))
    return W


def initialize(X, Y, k, type='uniform', random_state=0):
    if type == 'svd':
        U1, _, V1 = jnp.linalg.svd(X.T @ Y)
        U1 = U1[:, :k]
        V1 = V1[:, :k]
    elif type == 'uniform':
        U1 = jnp.ones((X.shape[1], k))
        V1 = jnp.ones((Y.shape[1], k))
        U1 = U1 / jnp.linalg.norm(U1, axis=0)
        V1 = V1 / jnp.linalg.norm(V1, axis=0)
    elif type == 'random':
        key = random.PRNGKey(random_state)
        key, subkey = random.split(key)
        U1 = random.normal(key, (X.shape[1], k))
        V1 = random.normal(subkey, (Y.shape[1], k))
        U1 = U1 / jnp.linalg.norm(U1, axis=0)
        V1 = V1 / jnp.linalg.norm(V1, axis=0)
    else:
        print(f'Initialization "{type}" not implemented')
        return
    return U1, V1


def initialize_gep(X, Y):
    n = X.shape[0]
    A = jnp.hstack((X, Y))
    A = jnp.dot(jnp.transpose(A), A) / n
    B = jsp.linalg.block_diag(jnp.dot(jnp.transpose(X), X), jnp.dot(jnp.transpose(Y), Y)) / n
    A = A - B
    return A, B
