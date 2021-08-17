# Importing necessary libraries
import jax.numpy as jnp
import numpy as np

# Calculate the eigenvalues of covariance matrix of X using Numpy for comparison
def calc_numpy(X,k):
    eigvals, eigvecs = jnp.linalg.eig(jnp.dot(jnp.transpose(X), X))
    idx = np.argsort(eigvals, axis=0)[::-1][:k]
    eigvecs = eigvecs[:, idx]
    eigvals = eigvals[idx]
    return eigvals, eigvecs
