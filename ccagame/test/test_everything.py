import jax.numpy as jnp
import numpy as np
from jax import random

from ccagame import cca, pls, pca
from ccagame.solver import gd_solve, agd_solve, svrg_solve, sgd_solve


def test_pca():
    """
    At the moment just checks they all run.

    Returns
    -------

    """
    n = 10
    p = 2
    q = 2
    n_components = 2
    batch_size = 2
    epochs = 10
    key = random.PRNGKey(0)
    X = random.normal(key, (n, p))
    X = X / jnp.linalg.norm(X, axis=0)
    numpy = pca.Numpy(n_components=n_components, epochs=epochs).fit(X)
    game = pca.Game(
        n_components=n_components, batch_size=batch_size, epochs=epochs
    ).fit(X)
    gha = pca.GHA(n_components=n_components, batch_size=batch_size, epochs=epochs).fit(
        X
    )
    oja = pca.Oja(n_components=n_components, batch_size=batch_size, epochs=epochs).fit(
        X
    )
    krasulina = pca.Krasulina(
        n_components=n_components, batch_size=batch_size, epochs=epochs
    ).fit(X)
    assert (
        np.testing.assert_almost_equal(
            [
                game.score(X),
                gha.score(X),
                oja.score(X),
                krasulina.score(X),
            ],
            numpy.score(X),
            decimal=0,
        )
        is None
    )


def test_cca():
    """
    At the moment just checks they all run.

    Returns
    -------

    """
    n = 10
    p = 2
    q = 2
    n_components = 2
    batch_size = 2
    epochs = 4
    lr=1e-1
    key = random.PRNGKey(0)
    key, subkey = random.split(key)
    X = random.normal(key, (n, p))
    X = X / jnp.linalg.norm(X, axis=0)
    Y = random.normal(subkey, (n, q))
    Y = Y / jnp.linalg.norm(Y, axis=0)
    numpy = cca.Numpy(n_components=n_components).fit(X, Y)
    ccalin = cca.CCALin(n_components=n_components, epochs=epochs).fit(X, Y)
    game = cca.Game(
        n_components=n_components, batch_size=batch_size, epochs=epochs, lr=lr
    ).fit(X, Y)
    msg = cca.MSG(n_components=n_components, batch_size=batch_size, epochs=epochs).fit(
        X, Y
    )
    lagrange = cca.Lagrange(
        n_components=n_components, batch_size=batch_size, epochs=epochs
    ).fit(X, Y)
    genoja = cca.Genoja(
        n_components=n_components, batch_size=batch_size, epochs=epochs
    ).fit(X, Y)
    assert (
        np.testing.assert_almost_equal(
            [
                ccalin.score(X, Y),
                game.score(X, Y),
                msg.score(X, Y),
                lagrange.score(X, Y),
                genoja.score(X, Y),
            ],
            numpy.score(X, Y),
            decimal=0,
        )
        is None
    )


def test_pls():
    """
    At the moment just checks they all run.

    Returns
    -------

    """
    n = 10
    p = 2
    q = 2
    n_components = 2
    batch_size = 2
    epochs = 4
    key = random.PRNGKey(0)
    key, subkey = random.split(key)
    X = random.normal(key, (n, p))
    X = X / jnp.linalg.norm(X, axis=0)
    Y = random.normal(subkey, (n, q))
    Y = Y / jnp.linalg.norm(Y, axis=0)
    numpy = pls.Numpy(n_components=n_components).fit(X, Y)
    game = pls.Game(
        n_components=n_components, batch_size=batch_size, epochs=epochs
    ).fit(X, Y)
    batch = pls.Batch(n_components=n_components, epochs=epochs).fit(X, Y)
    msg = pls.Incremental(n_components=n_components, epochs=epochs).fit(X, Y)
    lagrange = pls.MSG(
        n_components=n_components, batch_size=batch_size, epochs=epochs
    ).fit(X, Y)
    genoja = pls.SGD(
        n_components=n_components, batch_size=batch_size, epochs=epochs
    ).fit(X, Y)
    assert (
        np.testing.assert_almost_equal(
            [
                game.score(X, Y),
                batch.score(X, Y),
                msg.score(X, Y),
                lagrange.score(X, Y),
                genoja.score(X, Y),
            ],
            numpy.score(X, Y),
            decimal=0,
        )
        is None
    )

def test_solvers():
    import numpy as np

    np.random.seed(42)

    def ls(w, X, y):
        return (jnp.linalg.norm(X @ w - y) ** 2) / X.shape[0]

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
    assert (
        np.testing.assert_array_almost_equal(
            ls(w_gd, X, y), ls(w_exact, X, y), decimal=1
        )
        is None
    )
    assert (
        np.testing.assert_array_almost_equal(
            ls(w_agd, X, y), ls(w_exact, X, y), decimal=1
        )
        is None
    )
    assert (
        np.testing.assert_array_almost_equal(
            ls(w_sgd, X, y), ls(w_exact, X, y), decimal=1
        )
        is None
    )
    assert (
        np.testing.assert_array_almost_equal(
            ls(w_svrg, X, y), ls(w_exact, X, y), decimal=1
        )
        is None
    )
