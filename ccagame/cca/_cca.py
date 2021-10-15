from ccagame.pls import _PLS


class _CCA(_PLS):
    def __init__(self, n_components=2, *, scale=True, copy=True, wandb=True, verbose=False, random_state=None):
        super().__init__(n_components, scale=scale, copy=copy, wandb=wandb, verbose=verbose, random_state=random_state)

    def score(self, X, y, sample_weight=None):
        X, Y = self.transform(X, y)
        return self.TCC(X, Y)
