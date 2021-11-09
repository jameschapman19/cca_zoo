from ccagame.pls import _PLS
import jax.numpy as jnp
import jax.scipy as jsp


class _CCA(_PLS):
    def __init__(
        self,
        n_components=2,
        *,
        scale=True,
        copy=True,
        wandb=True,
        verbose=False,
        random_state=None
    ):
        super().__init__(
            n_components,
            scale=scale,
            copy=copy,
            wandb=wandb,
            verbose=verbose,
            random_state=random_state,
        )

    def score(self, X, y=None, sample_weight=None):
        X_hat, Y_hat = self.transform(X, y)
        return self.TCC(X_hat, Y_hat)

    @staticmethod
    def TCC(X, Y):
        dof = X.shape[0]
        C = jnp.hstack((X, Y))
        C = C.T @ C / dof
        # Get the block covariance matrix placing Xi^TX_i on the diagonal
        D = jsp.linalg.block_diag(*[m.T @ m for i, m in enumerate([X, Y])]) / dof
        C = C - jsp.linalg.block_diag(*[view.T @ view / dof for view in [X, Y]]) + D
        R = jnp.linalg.inv(jnp.linalg.cholesky(D))
        # In MCCA our eigenvalue problem Cv = lambda Dv
        C_whitened = R @ C @ R.T
        eigvals = jnp.linalg.eigvalsh(C_whitened)[::-1][: X.shape[1]] - 1
        return eigvals.real.sum()
