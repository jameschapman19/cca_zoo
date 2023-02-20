from typing import Iterable, Union

import numpy as np
import numpy.linalg as la
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KernelCenterer
from sklearn.utils.validation import check_is_fitted

from cca_zoo.models._iterative._gradkcca import GradKCCA
from cca_zoo.utils import _check_views, _process_parameter


class SCCA_HSIC(GradKCCA):
    """
    References
    ----------
    [1] Uurtio, V., Bhadra, S., Rousu, J. Sparse Non-Linear CCA through Hilbert-Schmidt Independence Criterion. IEEE International Conference on Data Mining (ICDM 2018), to appear

    """

    def __init__(
        self,
        latent_dims: int = 1,
        scale: bool = True,
        centre=True,
        copy_data=True,
        random_state=None,
        proj: Union[Iterable[float], float] = "l1",
        gamma: Iterable[float] = None,
        coef0: Iterable[float] = None,
        repetitions=5,
        initialization: Union[str, callable] = "random",
        nystrom=False,
        nystrom_components=100,
        c=1,
    ):
        super().__init__(
            latent_dims=latent_dims,
            scale=scale,
            centre=centre,
            copy_data=copy_data,
            random_state=random_state,
            initialization=initialization,
            proj=proj,
            gamma=gamma,
            coef0=coef0,
            kernel="rbf",
            repetitions=repetitions,
            nystrom=nystrom,
            nystrom_components=nystrom_components,
        )
        self.c = c

    def _check_params(self):
        self.proj = _process_parameter("proj", self.proj, None, self.n_views)
        self.kernel = _process_parameter("kernel", self.kernel, "linear", self.n_views)
        self.gamma = _process_parameter("gamma", self.gamma, None, self.n_views)
        self.coef0 = _process_parameter("coef0", self.coef0, 1, self.n_views)
        self.degree = _process_parameter("degree", self.degree, 1, self.n_views)
        self.c = _process_parameter("c", self.c, 1, self.n_views)

    def _objective(self, views, scores, weights) -> int:
        N = scores[0].shape[0]
        return np.trace(scores[0] @ scores[1]).sum() / (N - 1) ** 2

    def backracking_line_search(self, w, gw, stp, X, cKv, obj_old, view_index):
        while True:
            w_new = w + gw * stp
            if self.proj[view_index] == "l1":
                w_new = self._proj_l1(w_new, self.c[view_index])
            elif self.proj[view_index] == "l2":
                w_new = self._proj_l2(w_new, self.c[view_index])
            else:
                raise ValueError(
                    "projection {self.proj[view_index]} not supported. Pass a generator implementing this method"
                )
            Kw = self._get_kernel(view_index, X @ w_new[:, None])
            obj_new = self._objective(None, (Kw, cKv), None)
            if obj_new > obj_old + 1e-4 * np.abs(obj_old):
                return w_new
            elif stp < 1e-7:
                return w
            else:
                stp /= 2

    def _update(self, views, scores, weights):
        flipped_idxs = [1, 0]
        K = [
            self._get_kernel(i, view @ weights[i][:, None])
            for i, view in enumerate(views)
        ]
        cK = [KernelCenterer().fit_transform(K_) for K_ in K]
        weights = [
            self._proj_l2(w, self.c[view_index]) for view_index, w in enumerate(weights)
        ]
        for view_index, (view, flipped_idx) in enumerate(zip(views, flipped_idxs)):
            obj_old = self._objective(views, (K[view_index], cK[flipped_idx]), weights)
            grad = gradf_gauss_sgd(
                K[view_index],
                cK[flipped_idx],
                view,
                self.gamma[view_index],
                weights[view_index],
            )
            gamma = la.norm(grad)
            weights[view_index] = self.backracking_line_search(
                weights[view_index],
                grad,
                gamma,
                view,
                cK[flipped_idx],
                obj_old,
                view_index,
            )
            K[view_index] = self._get_kernel(
                view_index, view @ weights[view_index][:, None]
            )
            cK[view_index] = KernelCenterer().fit_transform(K[view_index])
        return K, weights

    def transform(self, views: Iterable[np.ndarray], **kwargs):
        """

        Parameters
        ----------
        views : list/tuple of numpy arrays or array likes with the same number of rows (samples)
        kwargs : any additional keyword arguments required by the given model

        Returns
        -------
        transformed_views : list of numpy arrays

        """
        check_is_fitted(self, attributes=["weights"])
        views = _check_views(
            *views
        )
        views = self._centre_scale_transform(views)
        transformed_views = []
        for i, (view) in enumerate(views):
            transformed_view = view @ self.weights[i]
            transformed_views.append(transformed_view)
        return transformed_views


def gradf_gauss_sgd(K1, cK2, X, a, u):
    if a is None:
        a = 1 / X.shape[1]
    N = K1.shape[0]
    temp = 0
    id1 = np.argsort(np.random.rand(N))[: int(N / 10)]
    id2 = np.argsort(np.random.rand(N))[: int(N / 10)]
    for i in id1:
        for j in id2:
            temp += (
                K1[i, j] * cK2[i, j] * np.outer(X[i, :] - X[j, :], X[i, :] - X[j, :])
            )
    return -(2 * a * u.T @ temp).T


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

    kernel1 = SCCA_HSIC(proj="l2").fit((Xtrain, Ytrain))
    print(f"Training Correlation: {kernel1.score((Xtrain, Ytrain))}")
    print(f"Test Correlation: {kernel1.score((Xtest, Ytest))}")

    plt.plot(*kernel1.transform((Xtrain, Ytrain)), "bo")
    plt.show()


if __name__ == "__main__":
    main()
