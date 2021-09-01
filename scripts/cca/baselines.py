# %%

import jax.numpy as jnp
from jax import random

# Imports
from ccagame.cca import calc_numpy, Numpy, Genoja, Game, Lagrange, AlternatingLeastSquares, CCALin

# %%

# Parameters
random_state = 0
n = 100
p = 10
q = 11
latent_dims = 5
max_iter = 300
batch_size = 10
epochs = 10
riemannian_projection = True
initialization = 'random'
lr = 1e-1
alpha = 100
beta_0 = 100

# %%

# Data Generation
key = random.PRNGKey(random_state)
key, subkey = random.split(key)
X = random.normal(key, (n, p))
X = X / jnp.linalg.norm(X, axis=0)
Y = random.normal(subkey, (n, q))
Y = Y / jnp.linalg.norm(Y, axis=0)

# %%

# Model
corr_sk, Usk, Vsk = calc_numpy(X, Y, k=latent_dims)
print("\n Eigenvalues calculated using scikit are :\n", corr_sk)
print("\n Sum :\n", jnp.sum(corr_sk))

numpy = Numpy(scale=False, n_components=latent_dims).fit(X, Y)

game = Game(scale=False, lr=lr, batch_size=batch_size, epochs=epochs, n_components=latent_dims, verbose=True,
            mu=True).fit(X, Y)

sgd = Genoja(scale=False, batch_size=batch_size, epochs=epochs, n_components=latent_dims, verbose=True, alpha=alpha,
             beta_0=beta_0).fit(X, Y)

batch = Lagrange(scale=False, lr=lr, epochs=epochs, n_components=latent_dims, verbose=True).fit(X, Y)

incremental = AlternatingLeastSquares(scale=False, lr=lr, epochs=epochs, n_components=latent_dims, verbose=True).fit(X,
                                                                                                                     Y)

ccalin = CCALin(scale=False, lr=lr, epochs=epochs, n_components=latent_dims, verbose=True).fit(X, Y)
