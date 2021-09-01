from ccagame.pls import _PLS


class _CCA(_PLS):
    def __init__(self, n_components=2, *, scale=True, copy=True):
        super().__init__(n_components, scale=scale, copy=copy)
