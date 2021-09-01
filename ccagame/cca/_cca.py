from ccagame.cca.utils import TCC
from ccagame.pls import _PLS


class _CCA(_PLS):
    def __init__(self, n_components=2, *, scale=True, copy=True):
        super().__init__(n_components, scale=scale, copy=copy)

    def score(self, X, y, sample_weight=None):
        return TCC(X, y, self.x_weights, self.y_weights)
