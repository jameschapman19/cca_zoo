import jax.numpy as jnp
from jax import grad
from jax import jit, vmap, random


def svrg_solve(fn, *args, x=None, in_axes=None, iters=1000, lr=1e-2, random_state=0, verbose=False):
    if in_axes is None:
        in_axes = tuple([None] + [0] * len(args))
    n = args[0].shape[0]
    key = random.PRNGKey(random_state)
    sample_grad = jit(vmap(grad(fn, argnums=0), in_axes=in_axes))
    fn_eval = jit(vmap(fn, in_axes=in_axes))
    for t in range(iters):
        previous = sample_grad(x, *args)
        for m_ in range(int(random.randint(key, [1], 0, n))):
            current = sample_grad(x, *args)
            i = random.randint(key, [1], 0, n)
            x = x - lr * (
                        jnp.squeeze(current[i], axis=0) - jnp.squeeze(previous[i], axis=0) + jnp.mean(previous, axis=0))
        if verbose:
            print(f'Value of function: {jnp.mean(fn_eval(x, *args, ))}')
    return x


def main():
    import jax.numpy as jnp
    import numpy as np
    np.random.seed(42)

    def ls(w, X, y):
        return jnp.linalg.norm(jnp.dot(X, w) - y)

    n = 100
    p = 10
    q = 2
    X = jnp.array(np.random.rand(n, p))
    X = X / jnp.linalg.norm(X, axis=0)
    y = jnp.array(np.random.rand(n, 1))
    y = y / jnp.linalg.norm(y, axis=0)
    w = jnp.array(np.random.rand(p, 1))

    w_ = svrg_solve(ls, X, y, x=w)

    print()


if __name__ == '__main__':
    main()
