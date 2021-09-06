from jax import random

from ccagame.pca import Game, GHA, Oja, Krasulina, Numpy

# Parameters
random_state = 0
n = 100
p = 10
latent_dims = 3
epochs = 50
riemannian_projection = False
lr = 1e-1
batch_size = 100
# This implements the version with unbiased updates analagous to Eigengame:Unloaded
mu = True

key = random.PRNGKey(0)
X = random.normal(key, (n, p))

numpy = Numpy(scale=False, n_components=latent_dims).fit(X)
print("\n Eigenvalues calculated using numpy are :\n", numpy.score(X))
print("\n Time :\n", numpy.fit_time)

game = Game(scale=False, lr=lr, batch_size=batch_size, epochs=epochs, n_components=latent_dims, verbose=True,
            mu=False).fit(X)
print("\n Eigenvalues calculated using game are :\n", game.score(X))
print("\n Time :\n", game.fit_time)

gha = GHA(scale=False, n_components=latent_dims, verbose=True, lr=lr).fit(X)
print("\n Eigenvalues calculated using gha are :\n", gha.score(X))
print("\n Time :\n", gha.fit_time)

oja = Oja(scale=False, n_components=latent_dims, verbose=True, lr=1e-2).fit(X)
print("\n Eigenvalues calculated using oja are :\n", oja.score(X))
print("\n Time :\n", oja.fit_time)

krasulina = Krasulina(scale=False, n_components=latent_dims, verbose=True, lr=1e-2).fit(X)
print("\n Eigenvalues calculated using krasulina are :\n", krasulina.score(X))
print("\n Time :\n", krasulina.fit_time)
