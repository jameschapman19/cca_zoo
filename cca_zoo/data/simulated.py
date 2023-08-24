import itertools
from typing import List, Union

import numpy as np
from scipy.linalg import block_diag
from sklearn.datasets import make_spd_matrix
from sklearn.utils.validation import check_random_state

from cca_zoo.utils import _process_parameter


class LinearSimulatedData:
    """
    Class for generating simulated data for linear model with multiple views.
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
            "view_sparsity", view_sparsity, 0.5, len(view_features)
        )
        self.structure = _process_parameter(
            "structure", structure, "random", len(view_features)
        )
        self.positive = _process_parameter(
            "positive", positive, False, len(view_features)
        )

        cov_matrices, self.true_features = self._generate_covariance_matrices()
        joint_cov = self._generate_joint_covariance(cov_matrices)
        self.chol = np.linalg.cholesky(joint_cov)

    def _generate_covariance_matrix(self, view_features, view_structure):
        """Generates a covariance matrix for a single view."""
        if view_structure == "identity":
            cov = np.eye(view_features)
        else:
            cov = make_spd_matrix(view_features)

        var = np.diag(cov)
        cov = cov / np.sqrt(var)
        cov = cov.T / np.sqrt(var)

        return cov

    def _generate_joint_covariance(self, cov_matrices):
        """Generates a joint covariance matrix for all views."""
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
        """Computes the cross-covariance matrix for a pair of views."""
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

    def _generate_covariance_matrices(self):
        """Generates a list of covariance matrices and true features for each view."""
        cov_matrices = [
            self._generate_covariance_matrix(f, s)
            for f, s in zip(self.view_features, self.structure)
        ]
        true_features = [
            self._generate_true_feature(cov, s, pos)
            for cov, s, pos in zip(cov_matrices, self.view_sparsity, self.positive)
        ]
        return cov_matrices, true_features

    def _generate_true_feature(self, cov, sparsity, is_positive):
        """Generates a true feature matrix for a single view."""
        weights = self._generate_weights(cov.shape[0])
        weights = self._apply_sparsity(weights, sparsity)

        if is_positive:
            weights = np.abs(weights)

        weights = self._decorrelate_weights(weights, cov)
        return weights / np.sqrt(np.diag(weights.T @ cov @ weights))

    def _generate_weights(self, view_features):
        return self.random_state.randn(view_features, self.latent_dims)

    def _apply_sparsity(self, weights, sparsity):
        if sparsity <= 1:
            sparsity = np.ceil(sparsity * weights.shape[0]).astype(int)

        mask = self._generate_sparsity_mask(weights.shape, sparsity)
        return weights * mask

    @staticmethod
    def _generate_sparsity_mask(shape, sparsity):
        mask_elements = [0] * (shape[0] - sparsity) + [1] * sparsity
        mask = np.stack([mask_elements] * shape[1]).T
        np.random.shuffle(mask)
        return mask.astype(bool)

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
