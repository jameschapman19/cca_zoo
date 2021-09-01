# %%

import jax.numpy as jnp
from jax import random

# Imports
from ccagame.cca import Numpy, Genoja, Game, Lagrange, AlternatingLeastSquares, CCALin

# %%

# Parameters
random_state = 0
n = 100
p = 10
q = 11
latent_dims = 5
max_iter = 300
batch_size = 100
epochs = 10
riemannian_projection = True
initialization = 'random'
lr = 1
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
numpy = Numpy(scale=False, n_components=latent_dims).fit(X, Y)
print("\n Eigenvalues calculated using numpy are :\n", numpy.score(X, Y))

lagrange = Lagrange(scale=False, lr=1e-3, epochs=epochs, n_components=latent_dims, verbose=True).fit(X, Y)
print("\n Eigenvalues calculated using numpy are :\n", lagrange.score(X, Y))

ccalin = CCALin(scale=False, epochs=epochs, n_components=latent_dims, verbose=True).fit(X, Y)
print("\n Eigenvalues calculated using CCALin are :\n", ccalin.score(X, Y))

game = Game(scale=False, lr=lr, batch_size=batch_size, epochs=epochs, n_components=latent_dims, verbose=True,
            mu=True).fit(X, Y)
print("\n Eigenvalues calculated using game are :\n", game.score(X, Y))

genoja = Genoja(scale=False, batch_size=batch_size, epochs=epochs, n_components=latent_dims, verbose=True, alpha=alpha,
                beta_0=beta_0).fit(X, Y)
print("\n Eigenvalues calculated using genoja are :\n", genoja.score(X, Y))

als = AlternatingLeastSquares(scale=False, lr=lr, epochs=epochs, n_components=latent_dims, verbose=True).fit(X, Y)
print("\n Eigenvalues calculated using ALS are :\n", als.score(X, Y))

