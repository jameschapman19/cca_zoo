# %%

import jax.numpy as jnp
from jax import random

# Imports
from ccagame.cca import Numpy, Genoja, Game, Lagrange, MSG, CCALin
from sklearn.cross_decomposition import CCA
import numpy as np

# %%

# Parameters
random_state = 0
n = 100
p = 10
q = 11
latent_dims = 5
batch_size = 5
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

a = CCA(n_components=5).fit(X, Y)
n = np.corrcoef(a.x_scores_, a.y_scores_, rowvar=False)
# Model
numpy = Numpy(scale=False, n_components=latent_dims).fit(X, Y)
print("\n Eigenvalues calculated using numpy are :\n", numpy.score(X, Y))
print("\n Time :\n", numpy.fit_time)

ccalin = CCALin(scale=False, epochs=epochs, n_components=latent_dims, verbose=True).fit(X, Y)
print("\n Eigenvalues calculated using game are :\n", ccalin.score(X, Y))
print("\n Time :\n", ccalin.fit_time)

game = Game(scale=False, lr=lr, batch_size=batch_size, epochs=epochs, n_components=latent_dims, verbose=True,
            mu=True, simultaneous=True).fit(X, Y)
print("\n Eigenvalues calculated using game are :\n", game.score(X, Y))
print("\n Time :\n", game.fit_time)

msg = MSG(scale=False, lr=lr,n_components=latent_dims,epochs=epochs,verbose=True).fit(X, Y)
print("\n Eigenvalues calculated using msg are :\n", msg.score(X, Y))
print("\n Time :\n", msg.fit_time)

lagrange = Lagrange(scale=False, lr=100, epochs=epochs, n_components=latent_dims, verbose=True).fit(X, Y)
print("\n Eigenvalues calculated using lagrange are :\n", lagrange.score(X, Y))
print("\n Time :\n", lagrange.fit_time)

genoja = Genoja(scale=False, batch_size=batch_size, epochs=epochs, n_components=latent_dims, verbose=True, alpha=alpha,
                beta_0=beta_0).fit(X, Y)
print("\n Eigenvalues calculated using genoja are :\n", genoja.score(X, Y))
print("\n Time :\n", genoja.fit_time)
