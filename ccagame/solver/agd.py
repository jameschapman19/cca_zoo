import jax.numpy as jnp
from jax import grad
from jax import jit, vmap


def print_objective(fn_eval, t, x, *args):
    val = jnp.mean(fn_eval(x, *args), axis=0)
    print(f'Iteration {t} : {val}')


# @partial(jit, static_argnums=(0), static_argnames=('in_axes', 'iterations', 'lr', 'mu', 'eps'))
def agd_solve(fn, *args, x=None, in_axes=None, iterations=1000, lr=1e-1, mu=0.0, verbose=False):
    if in_axes is None:
        in_axes = tuple([None] + [0] * len(args))
    sample_grad = jit(vmap(grad(fn, argnums=0), in_axes=in_axes))
    fn_eval = jit(vmap(fn, in_axes=in_axes))
    theta_ = jnp.zeros_like(x)
    for t in range(iterations):
        mu_grad = jnp.mean(sample_grad(x, *args), axis=0)
        theta = x - lr * mu_grad
        x = theta + mu * (theta - theta_)
        theta_ = theta
        if verbose:
            print_objective(fn_eval, t, x, *args)
    return x


def main():
    import numpy as np
    np.random.seed(42)

    def ls(w, X, y):
        return jnp.linalg.norm(jnp.dot(X, w) - y)

    n = 100
    p = 50
    X = jnp.array(np.random.rand(n, p))
    X = X / jnp.linalg.norm(X, axis=0)
    y = jnp.array(np.random.rand(n, 1))
    y = y / jnp.linalg.norm(y, axis=0)
    w = jnp.array(np.random.rand(p, 1))

    w_ = agd_solve(ls, X, y, x=w, mu=0.9, verbose=True)
    w_ = agd_solve(ls, X, y, x=w, mu=0.9, in_axes=(None, 0, 0), verbose=True)

    print()


if __name__ == '__main__':
    main()
