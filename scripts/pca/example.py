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
print("\n Eigenvalues calculated using numpy are :\n", numpy.score(X))

game = Game(scale=False, lr=lr, batch_size=batch_size, epochs=epochs, n_components=latent_dims, verbose=True,
            mu=True).fit(X)
print("\n Eigenvalues calculated using game are :\n", game.score(X))

gha = GHA(scale=False, n_components=latent_dims).fit(X)
print("\n Eigenvalues calculated using gha are :\n", gha.score(X))

oja = Oja(scale=False, n_components=latent_dims).fit(X)
print("\n Eigenvalues calculated using oja are :\n", oja.score(X))

krasulina = Krasulina(scale=False, n_components=latent_dims).fit(X)
print("\n Eigenvalues calculated using krasulina are :\n", krasulina.score(X))
