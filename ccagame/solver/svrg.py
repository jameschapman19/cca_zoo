from random import randint

import jax.numpy as jnp
from jax import grad
from jax import jit, vmap, random
from functools import partial


@partial(
    jit,
    static_argnums=(0),
    static_argnames=("in_axes", "iterations", "lr", "random_state"),
)
def svrg_solve(fn, X, y, x=None, in_axes=None, iterations=100, lr=1e-1, random_state=0):
    if in_axes is None:
        in_axes = tuple([None] + [0] * 2)
    sample_grad = jit(vmap(grad(fn, argnums=0), in_axes=in_axes))
    n = args[0].shape[0]
    key = random.PRNGKey(random_state)
    for t in range(iterations):
        previous = sample_grad(x, X, y)
        for m_ in range(randint(0, n)):
            current = sample_grad(x, X, y)
            i = random.randint(key, [1], 0, n)
            x = x - lr * (
                jnp.squeeze(current[i], axis=0)
                - jnp.squeeze(previous[i], axis=0)
                + jnp.mean(previous, axis=0)
            )
    return x


def main():
    import jax.numpy as jnp
    import numpy as np

    np.random.seed(42)

    def ls(w, X, y):
        return jnp.linalg.norm(jnp.dot(X, w) - y)

    n = 100
    p = 10
    X = jnp.array(np.random.rand(n, p))
    X = X / jnp.linalg.norm(X, axis=0)
    y = jnp.array(np.random.rand(n, 1))
    y = y / jnp.linalg.norm(y, axis=0)
    w = jnp.array(np.random.rand(p, 1))

    w_ = svrg_solve(ls, X, y, x=w, in_axes=(None, 0, 0))

    print()


if __name__ == "__main__":
    main()
