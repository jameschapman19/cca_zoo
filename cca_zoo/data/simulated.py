import itertools
from typing import List, Union

import numpy as np
from scipy import linalg
from scipy.linalg import block_diag
from sklearn.utils.validation import check_random_state

from cca_zoo.utils import _process_parameter


class LinearSimulatedData:
    def __init__(
        self,
        view_features: List[int],
        latent_dims: int = 1,
        view_sparsity: List[Union[int, float]] = None,
        correlation: Union[List[float], float] = 0.99,
        structure: Union[str, List[str]] = None,
        sigma: Union[List[float], float] = None,
        decay: float = 0.5,
        positive=None,
        random_state: Union[int, np.random.RandomState] = None,
    ):
        """

        Parameters
        ----------
        view_features : List[int]
            Number of features in each view
        latent_dims : int
            Number of latent dimensions
        view_sparsity : List[Union[int, float]]
            Sparsity of each view
        correlation : Union[List[float], float]
            Correlation between views
        structure : Union[str, List[str]]
            Structure of each view
        sigma : Union[List[float], float]
            Noise level of each view
        positive : None
            Unused
        random_state : Union[int, np.random.RandomState]
            Random state
        """
        self.view_features = view_features
        self.latent_dims = latent_dims
        self.correlation = correlation
        self.random_state = check_random_state(random_state)
        self.correlation = _process_parameter(
            "correlation", correlation, 0.99, latent_dims
        )
        # correlation must all be less than 1
        if np.any(np.abs(self.correlation) >= 1):
            raise ValueError("Magnitude of Correlation must be less than 1")
        self.structure = _process_parameter(
            "structure", structure, "identity", len(view_features)
        )
        self.view_sparsity = _process_parameter(
            "view_sparsity", view_sparsity, 1, len(view_features)
        )
        self.positive = _process_parameter(
            "positive", positive, False, len(view_features)
        )
        self.sigma = _process_parameter("sigma", sigma, 0.5, len(view_features))

        self.mean, covs, self.true_features = self._generate_covariance_matrices()
        joint_cov = self._generate_joint_covariance(covs)
        self.chol = np.linalg.cholesky(joint_cov)

    def _generate_covariance_matrix(self, view_p, view_structure, view_sigma):
        if view_structure == "identity":
            cov = np.eye(view_p)
        elif view_structure == "gaussian":
            cov = _generate_gaussian_cov(view_p, view_sigma)
        elif view_structure == "toeplitz":
            cov = _generate_toeplitz_cov(view_p, view_sigma)
        elif view_structure == "random":
            cov = _generate_random_cov(view_p, self.random_state)
        else:
            raise ValueError(f"Unknown structure {view_structure}")
        return cov

    def _generate_joint_covariance(self, covs):
        cov = block_diag(*covs)
        splits = np.concatenate(([0], np.cumsum(self.view_features)))
        for i, j in itertools.combinations(range(len(splits) - 1), 2):
            cross = np.zeros((self.view_features[i], self.view_features[j]))
            for _ in range(self.latent_dims):
                A = self.correlation[_] * np.outer(
                    self.true_features[i][:, _], self.true_features[j][:, _]
                )
                # Cross Bit
                cross += covs[i] @ A @ covs[j]
            cov[
                splits[i] : splits[i] + self.view_features[i],
                splits[j] : splits[j] + self.view_features[j],
            ] = cross
            cov[
                splits[j] : splits[j] + self.view_features[j],
                splits[i] : splits[i] + self.view_features[i],
            ] = cross.T
        return cov

    def _generate_covariance_matrices(self):
        mean = np.zeros(sum(self.view_features))
        covs = []
        true_features = []
        for view_p, sparsity, view_structure, view_positive, view_sigma in zip(
            self.view_features,
            self.view_sparsity,
            self.structure,
            self.positive,
            self.sigma,
        ):
            cov = self._generate_covariance_matrix(view_p, view_structure, view_sigma)
            weights = self.random_state.randn(view_p, self.latent_dims)
            if sparsity <= 1:
                sparsity = np.ceil(sparsity * view_p).astype("int")
            if sparsity < view_p:
                mask = np.stack(
                    (
                        np.concatenate(
                            ([0] * (view_p - sparsity), [1] * sparsity)
                        ).astype(bool),
                    )
                    * self.latent_dims,
                    axis=0,
                ).T
                mask = mask.flatten()
                self.random_state.shuffle(mask)
                mask = mask.reshape(weights.shape)
                weights = weights * mask
                if view_positive:
                    weights[weights < 0] = 0
            weights = _decorrelate_dims(weights, cov)
            weights /= np.sqrt(np.diag((weights.T @ cov @ weights)))
            true_features.append(weights)
            covs.append(cov)
        return mean, covs, true_features

    def sample(self, n: int):
        # check self.chol exists
        X = np.zeros((n, self.chol.shape[0]))
        for i in range(n):
            X[i, :] = self._chol_sample(self.mean, self.chol, self.random_state)
        views = np.split(X, np.cumsum(self.view_features)[:-1], axis=1)
        return views

    @staticmethod
    def _chol_sample(mean, chol, random_state):
        rng = check_random_state(random_state)
        return mean + chol @ rng.randn(mean.size)


def simple_simulated_data(
    n: int,
    view_features: List[int],
    view_sparsity: List[Union[int, float]] = None,
    eps: float = 0,
    transform=False,
    random_state=None,
):
    """
    Generate a simple simulated dataset with a single latent dimension

    Parameters
    ----------
    n : int
        Number of samples
    view_features :
        Number of features in each view
    view_sparsity : List[Union[int, float]]
        Sparsity of each view. If int, then number of non-zero features. If float, then proportion of non-zero features.
    eps : float
        Noise level
    transform : bool
        Whether to transform the data to be non-negative
    random_state : int, RandomState instance, default=None
        Controls the random seed in generating the data.

    Returns
    -------

    """
    random_state = check_random_state(random_state)
    z = random_state.randn(n)
    if transform:
        z = np.sin(z)
    views = []
    true_features = []
    view_sparsity = _process_parameter(
        "view_sparsity", view_sparsity, 0, len(view_features)
    )
    for p, sparsity in zip(view_features, view_sparsity):
        weights = random_state.normal(size=(p, 1))
        if sparsity <= 1:
            sparsity = np.ceil(sparsity * p).astype("int")
            weights[random_state.choice(np.arange(p), p - sparsity, replace=False)] = 0
        gaussian_x = random_state.normal(0, eps, size=(n, p)) * eps
        view = np.outer(z, weights)
        view += gaussian_x
        views.append(view / np.linalg.norm(view, axis=0))
        true_features.append(weights)
    return views, true_features


def _decorrelate_dims(up, cov):
    A = up.T @ cov @ up
    for k in range(1, A.shape[0]):
        up[:, k:] -= np.outer(up[:, k - 1], A[k - 1, k:] / A[k - 1, k - 1])
        A = up.T @ cov @ up
    return up


def _gaussian(x, mu, sig, dn):
    """
    Generate a gaussian covariance matrix

    :param x:
    :param mu:
    :param sig:
    :param dn:
    """
    return (
        np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))
        * dn
        / (np.sqrt(2 * np.pi) * sig)
    )


def _generate_gaussian_cov(p, sigma):
    x = np.linspace(-1, 1, p)
    x_tile = np.tile(x, (p, 1))
    mu_tile = np.transpose(x_tile)
    dn = 2 / (p - 1)
    cov = _gaussian(x_tile, mu_tile, sigma, dn)
    cov /= cov.max()
    return cov


def _generate_toeplitz_cov(p, sigma):
    c = np.arange(0, p)
    c = sigma**c
    cov = linalg.toeplitz(c, c)
    return cov


def _generate_random_cov(p, random_state):
    rng = check_random_state(random_state)
    cov_ = rng.randn(p, p)
    U, S, Vt = np.linalg.svd(cov_.T @ cov_)
    cov = U @ (1 + np.diag(rng.randn(p))) @ Vt
    return cov
