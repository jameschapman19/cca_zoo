import itertools
from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np
from scipy.linalg import block_diag
from sklearn.datasets import make_spd_matrix
from sklearn.utils.validation import check_random_state

from cca_zoo._utils import _process_parameter


class BaseData(ABC):
    def __init__(
        self,
        view_features: List[int],
        latent_dims: int = 1,
        random_state: Union[int, np.random.RandomState] = None,
    ):
        self.view_features = view_features
        self.latent_dims = latent_dims
        self.random_state = check_random_state(random_state)

    @abstractmethod
    def sample(self, n_samples: int):
        pass


class LatentVariableData(BaseData):
    def __init__(
        self,
        view_features: List[int],
        latent_dims: int = 1,
        random_state: Union[int, np.random.RandomState] = None,
        view_sparsity: Union[List[float], float] = None,
        positive: Union[bool, List[bool]] = False,
        structure="identity",
        signal_to_noise: float = 1.0,
    ):
        super().__init__(view_features, latent_dims, random_state)
        self.signal_to_noise = signal_to_noise
        self.view_sparsity = _process_parameter(
            "view_sparsity", view_sparsity, 1.0, len(view_features)
        )
        self.positive = _process_parameter(
            "positive", positive, False, len(view_features)
        )
        self.structure = _process_parameter(
            "structure", structure, "identity", len(view_features)
        )
        self.true_loadings = [
            self.generate_true_loading(view_features, view_sparsity, is_positive)
            for view_features, view_sparsity, is_positive in zip(
                self.view_features, self.view_sparsity, self.positive
            )
        ]
        self.cov_matrices = [
            self._generate_covariance_matrix(f, s)
            for f, s in zip(self.view_features, self.structure)
        ]
        self.true_features = [
            np.linalg.inv(cov) @ loading
            for cov, loading in zip(self.cov_matrices, self.true_loadings)
        ]

    def generate_true_loading(self, view_features, view_sparsity, is_positive):
        loadings = self.random_state.randn(view_features, self.latent_dims)
        if view_sparsity <= 1:
            view_sparsity = np.ceil(view_sparsity * loadings.shape[0]).astype(int)
        mask_elements = [0] * (loadings.shape[0] - view_sparsity) + [1] * view_sparsity
        mask = np.stack([mask_elements] * loadings.shape[1]).T
        np.random.shuffle(mask)
        loadings *= mask
        if is_positive:
            loadings = np.abs(loadings)
        loadings = loadings / np.sqrt(np.diag(loadings.T @ loadings))
        return loadings

    def _generate_covariance_matrix(self, view_features, view_structure):
        """Generates a covariance matrix for a single view."""
        if view_structure == "identity":
            cov = np.eye(view_features)
        else:
            cov = make_spd_matrix(view_features, random_state=self.random_state)
        # divide by sum of eigenvalues to normalize
        cov = cov * view_features / np.sum(np.linalg.eigvals(cov))
        return cov

    def sample(self, n_samples: int, return_latent: bool = False):
        random_latent = self.random_state.multivariate_normal(
            np.zeros(self.latent_dims), np.eye(self.latent_dims), n_samples
        )
        views = [
            random_latent @ true_loading.T
            + self.random_state.multivariate_normal(
                np.zeros(cov.shape[0]), cov, n_samples
            )
            / self.signal_to_noise
            for true_loading, cov in zip(self.true_loadings, self.cov_matrices)
        ]
        if return_latent:
            return views, random_latent
        return views

    @property
    def joint_cov(self):
        cov = np.zeros((sum(self.view_features), sum(self.view_features)))
        cov[: self.view_features[0], : self.view_features[0]] = (
            self.true_loadings[0] @ self.true_loadings[0].T + self.cov_matrices[0]
        )
        cov[self.view_features[0] :, self.view_features[0] :] = (
            self.true_loadings[1] @ self.true_loadings[1].T + self.cov_matrices[1]
        )
        cov[: self.view_features[0], self.view_features[0] :] = (
            self.true_loadings[0] @ self.true_loadings[1].T
        )
        cov[self.view_features[0] :, : self.view_features[0]] = (
            self.true_loadings[1] @ self.true_loadings[0].T
        )
        return cov


class JointData(BaseData):
    """
    Class for generating simulated data for a linear model with multiple representations.
    """

    def __init__(
        self,
        view_features: List[int],
        latent_dims: int = 1,
        view_sparsity: Union[List[float], float] = None,
        correlation: Union[List[float], float] = 0.99,
        structure: str = "random",
        positive: Union[bool, List[bool]] = False,
        random_state: Union[int, np.random.RandomState] = None,
    ):
        self.view_features = view_features
        self.latent_dims = latent_dims
        self.random_state = check_random_state(random_state)
        self.correlation = _process_parameter(
            "correlation", correlation, 0.99, latent_dims
        )
        self.view_sparsity = _process_parameter(
            "view_sparsity", view_sparsity, 1.0, len(view_features)
        )
        self.structure = _process_parameter(
            "structure", structure, "random", len(view_features)
        )
        self.positive = _process_parameter(
            "positive", positive, False, len(view_features)
        )
        cov_matrices = [
            self._generate_covariance_matrix(f, s)
            for f, s in zip(self.view_features, self.structure)
        ]
        self.true_features = [
            self.generate_true_weight(view_features, view_sparsity, is_positive, cov)
            for view_features, view_sparsity, is_positive, cov in zip(
                self.view_features, self.view_sparsity, self.positive, cov_matrices
            )
        ]
        self.true_loadings = [
            cov @ weight for weight, cov in zip(self.true_features, cov_matrices)
        ]
        self.joint_cov = self._generate_joint_covariance(cov_matrices)
        self.chol = np.linalg.cholesky(self.joint_cov)

    def generate_true_weight(self, view_features, view_sparsity, is_positive, cov):
        loadings = self.random_state.randn(view_features, self.latent_dims)
        if view_sparsity <= 1:
            view_sparsity = np.ceil(view_sparsity * loadings.shape[0]).astype(int)
        mask_elements = [0] * (loadings.shape[0] - view_sparsity) + [1] * view_sparsity
        mask = np.stack([mask_elements] * loadings.shape[1]).T
        np.random.shuffle(mask)
        loadings *= mask
        if is_positive:
            loadings = np.abs(loadings)
        loadings = self._decorrelate_weights(loadings, cov)
        return loadings / np.sqrt(np.diag(loadings.T @ cov @ loadings))

    def _generate_covariance_matrix(self, view_features, view_structure):
        """Generates a covariance matrix for a single view."""
        if view_structure == "identity":
            cov = np.eye(view_features)
        else:
            cov = make_spd_matrix(view_features, random_state=self.random_state)

        var = np.diag(cov)
        cov = cov / np.sqrt(var)
        cov = cov.T / np.sqrt(var)

        return cov

    def _generate_joint_covariance(self, cov_matrices):
        """Generates a joint covariance matrix for all representations."""
        joint_cov = block_diag(*cov_matrices)
        split_points = np.concatenate(([0], np.cumsum(self.view_features)))

        for i, j in itertools.combinations(range(len(split_points) - 1), 2):
            cross_cov = self._compute_cross_cov(cov_matrices, i, j)
            joint_cov[
                split_points[i] : split_points[i + 1],
                split_points[j] : split_points[j + 1],
            ] = cross_cov
            joint_cov[
                split_points[j] : split_points[j + 1],
                split_points[i] : split_points[i + 1],
            ] = cross_cov.T

        return joint_cov

    def _compute_cross_cov(self, cov_matrices, i, j):
        """Computes the cross-covariance matrix for a pair of representations."""
        cross_cov = np.zeros((self.view_features[i], self.view_features[j]))

        for _ in range(self.latent_dims):
            outer_product = np.outer(
                self.true_features[i][:, _], self.true_features[j][:, _]
            )
            cross_cov += (
                cov_matrices[i]
                @ (self.correlation[_] * outer_product)
                @ cov_matrices[j]
            )

        return cross_cov

    @staticmethod
    def _decorrelate_weights(weights, cov):
        product = weights.T @ cov @ weights
        for k in range(1, product.shape[0]):
            weights[:, k:] -= np.outer(
                weights[:, k - 1], product[k - 1, k:] / product[k - 1, k - 1]
            )
            product = weights.T @ cov @ weights
        return weights

    def sample(self, n_samples: int):
        random_data = self.random_state.randn(self.chol.shape[0], n_samples)
        samples = (self.chol @ random_data).T
        return np.split(samples, np.cumsum(self.view_features)[:-1], axis=1)
