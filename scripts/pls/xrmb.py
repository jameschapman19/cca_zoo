import jax.numpy as jnp

from ccagame.pls import calc_numpy, calc_sklearn, calc_game, calc_sgd, calc_incremental, calc_batch
# Parameters
from ccagame.utils import get_xrmb

random_state = 0
latent_dims = 3
epochs = 100
riemannian_projection = False
lr = 100
batch_size = 8

X, Y = get_xrmb(mode='Train')

# Model
corr_sk, Usk, Vsk = calc_sklearn(X, Y, k=latent_dims)
print("\n Eigenvalues calculated using scikit are :\n", corr_sk)
print("\n Sum :\n", jnp.sum(corr_sk))

corr_np, Unp, Vnp = calc_numpy(X, Y, k=latent_dims)
print("\n Eigenvalues calculated using numpy are :\n", corr_np)
print("\n Sum :\n", jnp.sum(corr_np))

corr, U, V = calc_game(X, Y, latent_dims, lr=lr, epochs=epochs,
                       riemannian_projection=riemannian_projection, random_state=random_state,
                       simultaneous=True, batch_size=batch_size)
print("\n Eigenvalues calculated using game are :\n", corr)
print("\n Sum :\n", jnp.sum(corr))

corr_b, Ub, Vb = calc_batch(X, Y, k=latent_dims, epochs=epochs)
print("\n Eigenvalues calculated using batch are :\n", corr_b)
print("\n Sum :\n", jnp.sum(corr_b))

corr_sg, Usg, Vsg = calc_sgd(X, Y, k=latent_dims, epochs=epochs, batch_size=batch_size)
print("\n Eigenvalues calculated using sgd are :\n", corr_sg)
print("\n Sum :\n", jnp.sum(corr_sg))

corr_inc, Uinc, Vinc = calc_incremental(X, Y, k=latent_dims, epochs=epochs)
print("\n Eigenvalues calculated using incremental are :\n", corr_inc)
print("\n Sum :\n", jnp.sum(corr_inc))
