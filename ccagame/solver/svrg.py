from random import randint

import jax.numpy as jnp
from jax import grad
from jax import jit, vmap


def svrg_solve(fn, *args, x=None, in_axes=None, iterations=1000, lr=1):
    if in_axes is None:
        in_axes = tuple([None] + [0] * 2)
    sample_grad = jit(vmap(grad(fn, argnums=0), in_axes=in_axes))
    n = args[0].shape[0]
    obj = []
    for epoch in range(iterations // n):
        previous = jnp.mean(sample_grad(x, *args), axis=0)
        for _ in range(n):
            i = randint(0, n)
            current = sample_grad(x, *args)[i]
            x = x - lr * (
                jnp.squeeze(current[i], axis=0)
                - jnp.squeeze(previous[i], axis=0)
                + previous
            )
            obj.append(float(fn(x, *args)))
    return x
