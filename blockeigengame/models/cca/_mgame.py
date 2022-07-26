from jax import jit

from ._game import Game


class MGame(Game):
    def __init__(self, mode, init_rng, config):
        super(MGame, self).__init__(mode, init_rng, config)

    @staticmethod
    @jit
    def _get_target(X, Y, U, V):
        Zx = X @ U.T
        Zy = Y @ V.T
        return Zx + Zy, Zx + zy
