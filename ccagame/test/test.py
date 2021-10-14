import jax.numpy as jnp

from ccagame.solver import gd_solve, agd_solve, svrg_solve, sgd_solve


def test_solvers():
    import numpy as np

    np.random.seed(42)

    def ls(w, X, y):
        return (jnp.linalg.norm(X @ w - y) ** 2)/X.shape[0]

    n = 100
    p = 50
    X = jnp.array(np.random.rand(n, p))
    X = X / jnp.linalg.norm(X, axis=0)
    y = jnp.array(np.random.rand(n, 1))
    y = y / jnp.linalg.norm(y, axis=0)
    w = jnp.array(np.random.rand(p, 1))

    w_gd = gd_solve(ls, X, y, x=w)
    w_agd = agd_solve(ls, X, y, x=w)
    w_svrg = svrg_solve(ls, X, y, x=w)
    w_sgd = sgd_solve(ls, X, y, x=w)
    w_exact = jnp.linalg.pinv(X) @ y
    print()
    assert np.testing.assert_array_almost_equal(ls(w_gd, X, y), ls(w_exact, X, y), decimal=1) is None
    assert np.testing.assert_array_almost_equal(ls(w_agd, X, y), ls(w_exact, X, y), decimal=1) is None
    assert np.testing.assert_array_almost_equal(ls(w_sgd, X, y), ls(w_exact, X, y), decimal=1) is None
    assert np.testing.assert_array_almost_equal(ls(w_svrg, X, y), ls(w_exact, X, y), decimal=1) is None
