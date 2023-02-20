from typing import Iterable, Union

import numpy as np
import numpy.linalg as la
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import pairwise_kernels
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

from cca_zoo.models._iterative._base import _BaseIterative
from cca_zoo.utils import _process_parameter, _check_views


class GradKCCA(_BaseIterative):
    """
    References
    ----------
    [1] Viivi Uurtio, Sahely Bhadra, and Juho Rousu. Large-scale sparse kernel canonical correlation analysis. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pages 6383–6391, Long Beach, California, USA, 09–15 Jun 2019. PMLR.

    """

    def __init__(
        self,
        latent_dims: int = 1,
        scale: bool = True,
        centre=True,
        copy_data=True,
        random_state=None,
        proj: Union[Iterable[float], float] = "l1",
        kernel: Iterable[Union[float, callable]] = None,
        gamma: Iterable[float] = None,
        degree: Iterable[float] = None,
        coef0: Iterable[float] = None,
        kernel_params: Iterable[dict] = None,
        repetitions=5,
        initialization: Union[str, callable] = "random",
        nystrom=False,
        nystrom_components=100,
    ):
        super().__init__(
            latent_dims=latent_dims,
            scale=scale,
            centre=centre,
            copy_data=copy_data,
            random_state=random_state,
            initialization=initialization,
        )
        self.proj = proj
        self.kernel_params = kernel_params
        self.gamma = gamma
        self.coef0 = coef0
        self.kernel = kernel
        self.degree = degree
        self.repetitions = repetitions
        self.nystrom = nystrom
        self.nystrom_components = nystrom_components

    def _check_params(self):
        self.proj = _process_parameter("proj", self.proj, "l1", self.n_views)
        self.kernel = _process_parameter("kernel", self.kernel, "linear", self.n_views)
        self.gamma = _process_parameter("gamma", self.gamma, None, self.n_views)
        self.coef0 = _process_parameter("coef0", self.coef0, 1, self.n_views)
        self.degree = _process_parameter("degree", self.degree, 1, self.n_views)
        self.nystrom_components = _process_parameter(
            "nystrom components", self.nystrom_components, 100, self.n_views
        )

    def _get_kernel(self, view_index, X, Y=None):
        if callable(self.kernel[view_index]):
            params = self.kernel_params[view_index] or {}
        else:
            params = {
                "gamma": self.gamma[view_index],
                "degree": self.degree[view_index],
                "coef0": self.coef0[view_index],
            }
        if self.nystrom:
            nystroem = Nystroem(
                kernel=self.kernel[view_index],
                n_components=self.nystrom_components[view_index],
                **params,
            )
            if Y is None:
                Y = X
            K = nystroem.fit_transform(X) @ nystroem.transform(Y).T
            return K
        else:
            return pairwise_kernels(
                X, Y, metric=self.kernel[view_index], filter_params=True, **params
            )

    def backracking_line_search(self, w, gw, stp, X, K, obj_old, view_index):
        while True:
            w_new = w + gw * stp
            w_new = w_new / la.norm(w_new)
            Kw_new = self._get_kernel(view_index, X, w_new[None, :])
            obj_new = self._objective(None, (Kw_new, K), None)
            if obj_new > obj_old + 1e-4 * np.abs(obj_old):
                w = w_new
                break
            elif stp < 1e-7:
                w = w_new
                break
            else:
                stp /= 2
        return w

    def _objective(self, views, scores, weights) -> int:
        return (
            scores[0].T
            @ scores[1]
            / (np.sqrt(scores[0].T @ scores[0]) * np.sqrt(scores[1].T @ scores[1]))
        )

    def _proj_l1(self, v, b):
        assert b > 0
        if la.norm(v, 1) < b:
            return v
        u = -np.sort(-np.abs(v))
        sv = np.cumsum(u)
        r = np.where(u > (sv - b) / np.arange(1, u.shape[0] + 1))
        if len(r[-1]) > 0:
            rho = r[-1][-1]
            tau = (sv[rho] - b) / rho
            theta = np.maximum(0, tau)
            return np.sign(v) * np.maximum(np.abs(v) - theta, 0)
        else:
            return v

    def _proj_l2(self, v, b):
        return v / la.norm(v) / b

    def _update(self, views, scores, weights):
        scores = [
            self._get_kernel(i, view, weights[i][None, :])
            for i, view in enumerate(views)
        ]
        obj_old = self._objective(views, scores, weights)
        for view_index, view in enumerate(views):
            # TODO gradients
            grad = 0
            gamma = la.norm(grad)
            weights[view_index] = self.backracking_line_search(
                weights[view_index],
                grad,
                gamma,
                view,
                scores[view_index],
                obj_old,
                view_index,
            )
            if self.proj[view_index] == "l1":
                weights[view_index] = self._proj_l1(
                    weights[view_index], self.c[view_index]
                )
            elif self.proj[view_index] == "l2":
                weights[view_index] = self._proj_l2(
                    weights[view_index], self.c[view_index]
                )
            else:
                raise ValueError(
                    "projection {self.proj[view_index]} not supported. Pass a generator implementing this method"
                )
        return scores, weights

    def transform(self, views: Iterable[np.ndarray], **kwargs):
        check_is_fitted(self, attributes=["weights"])
        views = _check_views(
            *views
        )
        views = self._centre_scale_transform(views)
        scores = [
            self._get_kernel(i, view, self.weights[i].T) for i, view in enumerate(views)
        ]
        return scores


def generate_data(n, p, q):
    X = np.random.uniform(-1, 1, [n, p])
    Y = np.random.uniform(-1, 1, [n, q])
    Y[:, 2] = X[:, 2] + X[:, 3] - Y[:, 3] + np.random.normal(0, 0.05, n)
    # Y[:,2] = np.power(X[:,2] + X[:,3],3) - Y[:,3] + np.random.normal(0,0.05,n)
    # Y[:,4] = np.exp(X[:,4] + X[:,5]) - Y[:,5] + np.random.normal(0,0.05,n)
    return X, Y


def main():
    import matplotlib.pyplot as plt

    np.set_printoptions(precision=2)

    X, Y = generate_data(1000, 8, 8)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y)

    kernel1 = GradKCCA().fit((Xtrain, Ytrain))
    print(f"Training Correlation: {kernel1.score((Xtrain, Ytrain))}")
    print(f"Test Correlation: {kernel1.score((Xtest, Ytest))}")

    plt.plot(*kernel1.transform((Xtrain, Ytrain)), "bo")
    plt.show()


if __name__ == "__main__":
    main()
