# Importing necessary libraries
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import random
from sklearn.cross_decomposition import CCA


# Calculate the eigenvalues of covariance matrix of X using Numpy for comparison
def calc_numpy_eig(X, Y, n, r=0):
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
    return (eigvals, eigvecs[:X.shape[1]], eigvecs[X.shape[1]:])


# Define utlity function, we will take grad of this in the
# update step, v is the current eigenvector being calculated
# X is the design matrix and V1 holds the previously computed eigenvectors
def model(u, v, X, Y, U1, k):
    C_xy = jnp.dot(jnp.transpose(X), Y)
    C_xx = jnp.dot(jnp.transpose(X), X)
    rewards = jnp.dot(jnp.transpose(u), jnp.dot(C_xy, v)) / (
                jnp.sqrt(jnp.dot(jnp.transpose(u), jnp.dot(C_xx, u))) * jnp.sqrt(
            jnp.dot(jnp.transpose(u), jnp.dot(C_xx, u))))
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
    du = jax.grad(model)(u, v, X, Y, U1, k)
    dv = jax.grad(model)(v, u, Y, X, V1, k)
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
def calc_eigengame_eigenvectors(X, Y, n, lr=1e-1, iterations=100, riemannian_projection=False, initialization='random',
                                random_state=0):
    if initialization == 'svd':
        U1, _, V1 = jnp.linalg.svd(X.T @ Y)
        U1 = U1[:, :n]
        V1 = V1[:, :n]
    elif initialization == 'random':
        key = random.PRNGKey(random_state)
        key, subkey = random.split(key)
        U1 = random.normal(key, (X.shape[1], n))
        V1 = random.normal(subkey, (Y.shape[1], n))
    else:
        print(f'Initialization "{initialization}" not implemented')
    for i in range(iterations):
        for k in range(n):
            u, v = update(U1[:, k], V1[:, k], X, Y, U1, V1, k, lr=lr, riemannian_projection=riemannian_projection)
            U1 = U1.at[:, k].set(u)
            V1 = V1.at[:, k].set(v)
        print(f'iteration {i}: {calc_eigengame_eigenvalues(X, Y, U1, V1)}')
    return U1, V1


# Calculate eigenvalues once the eigenvectors have been computed
def calc_eigengame_eigenvalues(X, Y, U1, V1):
    C_xy = jnp.dot(jnp.transpose(X), Y)
    C_xx = jnp.dot(jnp.transpose(X), X)
    C_yy = jnp.dot(jnp.transpose(Y), Y)
    n = jnp.size(V1, axis=1)
    eigvals = np.zeros((1, n))
    for k in range(n):
        eigvals[:, k] = jnp.dot(U1[:, k], jnp.dot(C_xy, V1[:, k].reshape(-1, 1))) / (
                jnp.sqrt(jnp.dot(U1[:, k], jnp.dot(C_xx, U1[:, k].reshape(-1, 1)))) * jnp.sqrt(
            jnp.dot(V1[:, k], jnp.dot(C_yy,
                                      V1[:, k].reshape(
                                          -1, 1)))))
    return eigvals


random_state = 0
key = random.PRNGKey(random_state)
key, subkey = random.split(key)
p = 4
q = 5
X = random.normal(key, (20, p))
Y = random.normal(subkey, (20, q))
# Y=jnp.array(X)
Xnp = np.array(X)
Ynp = np.array(Y)

latent_dims = 4
max_iter = 200
riemannian_projection = True
lr = 1e-1
cca = CCA(n_components=p, scale=False).fit(Xnp, Ynp)
ccax, ccay = cca.transform(Xnp, Ynp)
cca_corr = np.diag(np.corrcoef(ccax, ccay, rowvar=False)[p:, :p])
p, U1np, V1np = calc_numpy_eig(X, Y, n=latent_dims)
U1, V1 = calc_eigengame_eigenvectors(X, Y, latent_dims, lr=lr, iterations=max_iter,
                                     riemannian_projection=riemannian_projection, random_state=random_state)
print("\n Eigenvalues calculated using numpy are :\n", p)
print("\n Eigenvalues calculate using the Eigengame are :\n", calc_eigengame_eigenvalues(X, Y, U1, V1))
print("\n Left Eigenvectors calculated using numpy are :\n", U1np)
print("\n Left Eigenvectors calculated using the Eigengame are :\n", U1)
print("\n Right Eigenvectors calculated using numpy are :\n", V1np)
print("\n Right Eigenvectors calculated using the Eigengame are :\n", V1)
print("\n Squared error in estimation of eigenvectors as compared to numpy :\n",
      np.sum((np.abs(U1np) - np.abs(U1)) ** 2, axis=0))
