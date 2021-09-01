# %%

import jax.numpy as jnp
from jax import random

from ccagame.pls import Game, SGD, Incremental, Batch, Numpy
# Imports
from ccagame.pls import calc_numpy

# %%

# Parameters
random_state = 0
n = 100
p = 10
q = 11
latent_dims = 3
epochs = 100
riemannian_projection = False
lr = 1e-2
batch_size = 8

# %%

# Data Generation
key = random.PRNGKey(random_state)
key, subkey = random.split(key)
X = random.normal(key, (n, p))
Y = random.normal(subkey, (n, q))

# Model

corr_np, Unp, Vnp = calc_numpy(X, Y, k=latent_dims)
print("\n Eigenvalues calculated using numpy are :\n", corr_np)
print("\n Sum :\n", jnp.sum(corr_np))

numpy = Numpy(scale=False, n_components=latent_dims).fit(X, Y)

game = Game(scale=False, lr=lr, batch_size=batch_size, epochs=epochs, n_components=latent_dims, verbose=True,
            mu=True).fit(X, Y)

sgd = SGD(scale=False, lr=lr, batch_size=batch_size, epochs=epochs, n_components=latent_dims, verbose=True).fit(X, Y)

batch = Batch(scale=False, lr=lr, epochs=epochs, n_components=latent_dims, verbose=True).fit(X, Y)

incremental = Incremental(scale=False, lr=lr, epochs=epochs, n_components=latent_dims, verbose=True).fit(X, Y)
