# %%

import jax.numpy as jnp
from jax import random

from ccagame.pls import Game, SGD, Incremental, Batch, Numpy
# Imports
from ccagame.pls import calc_numpy
import time

# %%

# Parameters
random_state = 0
n = 100
p = 10
q = 11
latent_dims = 3
epochs = 10
riemannian_projection = False
lr = 1e-2
batch_size = 10

# %%

# Data Generation
key = random.PRNGKey(random_state)
key, subkey = random.split(key)
X = random.normal(key, (n, p))
Y = random.normal(subkey, (n, q))

# Model

before = time.time()
numpy = Numpy(scale=False, n_components=latent_dims).fit(X, Y)
print("\n Eigenvalues calculated using numpy are :\n", numpy.score(X, Y))
print("\n Time :\n", time.time() - before)

before = time.time()
game = Game(scale=False, lr=lr, batch_size=batch_size, epochs=epochs, n_components=latent_dims, verbose=True,
            mu=True).fit(X, Y)
print("\n Eigenvalues calculated using game are :\n", game.score(X, Y))
print("\n Time :\n", time.time() - before)

before = time.time()
sgd = SGD(scale=False, lr=lr, batch_size=batch_size, epochs=epochs, n_components=latent_dims, verbose=True).fit(X, Y)
print("\n Eigenvalues calculated using sgd are :\n", sgd.score(X, Y))
print("\n Time :\n", time.time() - before)

before = time.time()
batch = Batch(scale=False, lr=lr, epochs=epochs, n_components=latent_dims, verbose=True).fit(X, Y)
print("\n Eigenvalues calculated using batch are :\n", batch.score(X, Y))
print("\n Time :\n", time.time() - before)

before = time.time()
incremental = Incremental(scale=False, lr=lr, epochs=epochs, n_components=latent_dims, verbose=True).fit(X, Y)
print("\n Eigenvalues calculated using incremental are :\n", incremental.score(X, Y))
print("\n Time :\n", time.time() - before)
