import time

from jax import random

from ccagame.pca import Game, GHA, Oja, Krasulina, Numpy

n = 100
p = 10
latent_dims = 3
batch_size = 8
epochs = 20

key = random.PRNGKey(0)
X = random.normal(key, (n, p))

numpy = Numpy(scale=False, n_components=latent_dims).fit(X)
print("\n Eigenvalues calculated using numpy are :\n", numpy.score(X))
print("\n Time :\n", numpy.fit_time)

game = Game(scale=False, lr=1e-1, batch_size=batch_size, epochs=epochs, n_components=latent_dims, verbose=True,
            mu=True).fit(X)
print("\n Eigenvalues calculated using game are :\n", game.score(X))
print("\n Time :\n", game.fit_time)


gha = GHA(scale=False, n_components=latent_dims, verbose=True,lr=1e-2).fit(X)
print("\n Eigenvalues calculated using gha are :\n", gha.score(X))
print("\n Time :\n", gha.fit_time)

oja = Oja(scale=False, n_components=latent_dims, verbose=True, lr=1e-2).fit(X)
print("\n Eigenvalues calculated using oja are :\n", oja.score(X))
print("\n Time :\n", oja.fit_time)

krasulina = Krasulina(scale=False, n_components=latent_dims, verbose=True, lr=1e-2).fit(X)
print("\n Eigenvalues calculated using krasulina are :\n", krasulina.score(X))
print("\n Time :\n", krasulina.fit_time)
