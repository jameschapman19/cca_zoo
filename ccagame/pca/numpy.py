# Importing necessary libraries
import jax.numpy as jnp


# Calculate the eigenvalues of covariance matrix of X using Numpy for comparison
def calc_numpy(X):
    p, q = jnp.linalg.eig(jnp.dot(jnp.transpose(X), X))
    return p, q
