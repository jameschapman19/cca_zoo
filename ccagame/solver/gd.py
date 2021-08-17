import jax.numpy as jnp
from jax import grad, vmap, jit


def gd_solve(fn, *args, x=None, in_axes=None, iters=1000, lr=1e-1, verbose=False):
    if in_axes is None:
        sample_grad = jit(grad(fn, argnums=0))
        fn_eval = jit(fn)
    else:
        sample_grad = jit(vmap(grad(fn, argnums=0), in_axes=in_axes))
        fn_eval = jit(vmap(fn, in_axes=in_axes))
    sample_grad = jit(vmap(grad(fn, argnums=0), in_axes=in_axes))
    fn_eval = jit(vmap(fn, in_axes=in_axes))
    for t in range(iters):
        mu_grad = jnp.mean(sample_grad(x, *args), axis=0)
        x = x - lr * mu_grad
        if verbose:
            print(f'Value of function: {jnp.mean(fn_eval(x, *args, ))}')
    return x


def main():
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

    w_ = gd_solve(ls, X, y, x=w, in_axes=(None,0,0))

    print()


if __name__ == '__main__':
    main()
