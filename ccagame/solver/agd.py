import jax.numpy as jnp
from jax import grad
from jax import jit


def agd_solve(fn, *args, x=None, iterations=1000, lr=1, Q=1e-1):
    sample_grad = jit(grad(fn, argnums=0))
    y_ = jnp.zeros_like(x)
    obj = []
    for _ in range(iterations):
        y = x - lr * sample_grad(x, *args)
        x = y + (Q ** 0.5 - 1) / (Q ** 0.5 + 1) * (y - y_)
        y_ = y
        obj.append(float(fn(x, *args)))
    return y
