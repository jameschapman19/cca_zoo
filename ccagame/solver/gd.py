from jax import grad, jit


def gd_solve(fn, *args, x=None, iterations=1000, lr=1):
    sample_grad = jit(grad(fn, argnums=0))
    obj = []
    for t in range(iterations):
        x = x - lr * sample_grad(x, *args)
        obj.append(float(fn(x, *args)))
    return x
