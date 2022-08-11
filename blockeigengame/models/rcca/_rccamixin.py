import jax.numpy as jnp
from cca_zoo.models import MCCA
from jax import jit

from blockeigengame.data_utils.mediamill import mediamill_true
from blockeigengame.data_utils.xrmb import xrmb_true
from blockeigengame.metrics import _correct_eigenvector_streak, _sum_cosine_similarities


class _RCCAMixin:
    def _init_ground_truth(self):
        if self.config.data == "xrmb":
            self.correct_U, self.correct_V = xrmb_true(cca=True)
            self.correct_U = self.correct_U[:, : self.config.n_components]
            self.correct_V = self.correct_V[:, : self.config.n_components]
        elif self.config.data == "mediamill":
            self.correct_U, self.correct_V = mediamill_true(cca=True)
            self.correct_U = self.correct_U[:, : self.config.n_components]
            self.correct_V = self.correct_V[:, : self.config.n_components]
        else:
            cca = MCCA(latent_dims=self.config.n_components, c=self.config.tau).fit((self.X, self.Y))
            self.correct_U, self.correct_V = cca.weights
        self.TCC_train = _TCC(self.X, self.Y, self.correct_U.T, self.correct_V.T)
        self.TCC_val = _TCC(self.X_val, self.Y_val, self.correct_U.T, self.correct_V.T)

    def _get_scalars(self, global_step):
        scalars = {}
        if global_step == 0 or (global_step + 1) % self.config.val_interval == 0:
            scalars["TCC train"] = _TCC(
                self.X, self.Y, self._U, self._V, self.config.tau[0], self.config.tau[1]
            )
            scalars["TCC val"] = _TCC(self.X_val, self.Y_val, self._U, self._V)
            scalars["PCC train"] = scalars["TCC train"] / self.TCC_train
            scalars["PCC val"] = scalars["TCC val"] / self.TCC_val
            scalars["correct x"] = _correct_eigenvector_streak(self._U, self.correct_U)
            scalars["correct y"] = _correct_eigenvector_streak(self._V, self.correct_V)
            scalars["sum cosine similarities x"] = _sum_cosine_similarities(
                self._U, self.correct_U
            )
            scalars["sum cosine similarities y"] = _sum_cosine_similarities(
                self._V, self.correct_V
            )
            return scalars

    def evaluate(self, global_step, rng):
        scalars = {}
        scalars["TCC train"] = _TCC(self.X, self.Y, self._U, self._V)
        scalars["TCC val"] = _TCC(self.X_val, self.Y_val, self._U, self._V)
        scalars["PCC train"] = scalars["TCC train"] / self.TCC_train
        scalars["PCC val"] = scalars["TCC val"] / self.TCC_val
        scalars["correct x"] = _correct_eigenvector_streak(self._U, self.correct_U)
        scalars["correct y"] = _correct_eigenvector_streak(self._V, self.correct_V)
        scalars["sum cosine similarities x"] = _sum_cosine_similarities(
            self._U, self.correct_U
        )
        scalars["sum cosine similarities y"] = _sum_cosine_similarities(
            self._V, self.correct_V
        )
        return scalars


@partial(jit, backend="cpu")
def _TCC(X, Y, U, V, tau_x, tau_y):
    Zx = X @ U.T
    Zy = Y @ V.T
    n = X.shape[0]
    Ux, Sx, Vx = jnp.linalg.svd(Zx, full_matrices=False)
    Uy, Sy, Vy = jnp.linalg.svd(Zy, full_matrices=False)
    Rx = Ux @ jnp.diag(Sx)
    Ry = Uy @ jnp.diag(Sy)
    Bx = tau_x + (1 - tau_x) * Sx ** 2
    By = tau_y + (1 - tau_y) * Sy ** 2
    Rxy = Rx.T @ Ry
    M = (
            jnp.diag(1 / jnp.sqrt(Bx / n))
            @ Rxy.T
            @ jnp.diag(1 / (Bx / n))
            @ Rxy
            @ jnp.diag(1 / jnp.sqrt(By / n))
    )
    eigvals, eigvecs = jnp.linalg.eigh(M)
    w_y = Vy.T @ jnp.diag(1 / jnp.sqrt(By / n)) @ eigvecs
    w_x = (
            Vx.T
            @ jnp.diag(1 / (Sx ** 2 / n))
            @ Rxy
            @ jnp.diag(1 / jnp.sqrt((By / n)))
            @ eigvecs
            / jnp.sqrt(eigvals)
    )
    return (jnp.linalg.eigvalsh(w_x.T @ Zx.T @ Zy @ w_y) / n).sum()
