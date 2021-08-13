# Importing necessary libraries
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np


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
    eigvals = eigvals[idx]
    return eigvals, eigvecs[:X.shape[1]], eigvecs[X.shape[1]:]
