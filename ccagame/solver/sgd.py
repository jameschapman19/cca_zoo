from random import randint

import jax.numpy as jnp
from jax import grad, vmap, jit


def sgd_solve(fn, *args, x=None, in_axes=None, iterations=1000, lr=1):
    if in_axes is None:
        in_axes = tuple([None] + [0] * len(args))
    sample_grad = jit(vmap(grad(fn, argnums=0), in_axes=in_axes))
    n = args[0].shape[0]
    obj = []
    for _ in range(iterations):
        i = randint(0, n)
        x = x - lr * sample_grad(x, *args)[i]
        obj.append(float(fn(x, *args)))
    return x


def main():
    import numpy as np

    np.random.seed(42)

    def ls(w, X, y):
        return jnp.linalg.norm(X @ w - y) ** 2

    n = 100
    p = 50
    X = jnp.array(np.random.rand(n, p))
    X = X / jnp.linalg.norm(X, axis=0)
    y = jnp.array(np.random.rand(n, 1))
    y = y / jnp.linalg.norm(y, axis=0)
    w = jnp.array(np.random.rand(p, 1))

    w_sgd, obj = sgd_solve(ls, X, y, x=w)
    w_exact = jnp.linalg.pinv(X) @ y
    print()


if __name__ == '__main__':
    main()
