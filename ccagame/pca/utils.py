import jax.numpy as jnp


# Calculate eigenvalues once the eigenvectors have been computed
def calc_eigenvalues(X, V1):
    M = jnp.dot(jnp.transpose(X), X)
    n = jnp.size(V1, axis=1)
    eigvals = jnp.zeros((1, n))
    for k in range(n):
        eigvals = eigvals.at[:, k].set(jnp.dot(V1[:, k], jnp.dot(M, V1[:, k].reshape(-1, 1))))
    return eigvals
