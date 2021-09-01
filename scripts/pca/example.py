import jax.numpy as jnp
from jax import random

from ccagame.pca import Game, GHA, Oja, Krasulina, Numpy

n = 100
p = 10
latent_dims = 3
lr = 1
batch_size = 8
epochs = 20

key = random.PRNGKey(0)
X = random.normal(key, (n, p))

numpy = Numpy(scale=False, n_components=latent_dims).fit(X)

game = Game(scale=False, lr=lr, batch_size=batch_size, epochs=epochs, n_components=latent_dims, verbose=True,
            mu=True).fit(X)

gha = GHA(scale=False, n_components=latent_dims).fit(X)
Oja = Oja(scale=False, n_components=latent_dims).fit(X)
Krasulina = Krasulina(scale=False, n_components=latent_dims).fit(X)
