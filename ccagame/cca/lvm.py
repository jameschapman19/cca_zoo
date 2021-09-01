# Importing necessary libraries
from functools import partial

import jax.numpy as jnp
from jax import random, grad, jit

from ccagame.cca.utils import initialize, TCC


@partial(jit)
def forward_model(U_, V_, Z, X, Y):
    X_hat = Z @ U_
    Y_hat = Z @ V_
    return -jnp.mean(jnp.sum((X_hat - X) ** 2, axis=1) + jnp.sum((Y_hat - Y) ** 2, axis=1))


@partial(jit)
def backward_model(U, V, Z, X, Y):
    Z_hat_X = X @ U
    Z_hat_Y = Y @ V
    return -jnp.mean(jnp.sum((Z_hat_X - Z) ** 2, axis=1) + jnp.sum((Z_hat_Y - Z) ** 2, axis=1))


# Update rule to be used for calculating eigenvectors
# @partial(jit,static_argnames=('lr'))
def update(U, V, U_, V_, Z, X, Y, lr=1):
    f_loss = forward_model(U_, V_, Z, X, Y)
    du_, dv_, dz = grad(forward_model, argnums=(0, 1, 2))(U_, V_, Z, X, Y)
    U_ = U_ + lr * du_
    V_ = V_ + lr * dv_
    Z = Z + lr * dz
    Z = Z / jnp.linalg.norm(Z, axis=0)
    b_loss = backward_model(U, V, Z, X, Y)
    du, dv = grad(backward_model, argnums=(0, 1))(U, V, Z, X, Y)
    U = U + lr * du
    V = V + lr * dv
    return U, V, U_, V_, Z


# Run the update step iteratively across all eigenvectors
def calc_lvm(X, Y, k, iterations=100,
             lr=1, random_state=0):
    n = X.shape[0]
    key = random.PRNGKey(random_state)
    Z = random.normal(key, (n, k))
    Z = Z / jnp.linalg.norm(Z, axis=0)
    U, V = initialize(X, Y, k, 'random', random_state)
    key = random.PRNGKey(random_state)
    key, subkey = random.split(key)
    U_ = random.normal(key, (k, X.shape[1]))
    V_ = random.normal(subkey, (k, Y.shape[1]))
    for i in range(iterations):
        U, V, U_, V_, Z, = update(U, V, U_, V_, Z, X, Y, lr=lr)
        print(f'iteration {i}: {TCC(X, Y, U, V)}')
    return TCC(X, Y, U, V), U, V
