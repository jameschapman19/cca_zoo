import itertools
from typing import List, Union

import numpy as np
from scipy.linalg import block_diag
from sklearn.datasets import make_spd_matrix
from sklearn.utils.validation import check_random_state

from cca_zoo.utils import _process_parameter


class LinearSimulatedData:
    # This class generates simulated data for linear classical with multiple views
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
        self.correlation = correlation
        self.random_state = check_random_state(random_state)
        # Process the correlation parameter to ensure it is a list of length latent_dimensions
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
        # Generate the covariance matrices and the true features for each view
        covs, self.true_features = self._generate_covariance_matrices()
        # Generate the joint covariance matrix by combining the view covariances and the cross-covariances
        joint_cov = self._generate_joint_covariance(covs)
        # Compute the Cholesky decomposition of the joint covariance matrix
        self.chol = np.linalg.cholesky(joint_cov)

    def _generate_covariance_matrix(self, view_features, view_structure):
        # Generate a covariance matrix for a single view based on its structure
        if view_structure == "identity":
            # Use an identity matrix as the covariance matrix
            cov = np.eye(view_features)
        elif view_structure == "random":
            # Use a random positive definite matrix as the covariance matrix
            cov = make_spd_matrix(view_features)
        return cov

    def _generate_joint_covariance(self, covs):
        # Generate a joint covariance matrix for all views by stacking the view covariances and adding cross-covariances
        cov = block_diag(*covs)
        splits = np.concatenate(([0], np.cumsum(self.view_features)))
        for i, j in itertools.combinations(range(len(splits) - 1), 2):
            cross = np.zeros((self.view_features[i], self.view_features[j]))
            for _ in range(self.latent_dims):
                # Compute the cross-covariance matrix for a pair of views and a latent dimension using the correlation coefficient and the true features
                A = self.correlation[_] * np.outer(
                    self.true_features[i][:, _], self.true_features[j][:, _]
                )
                # Multiply the cross-covariance matrix by the view covariances to get the final cross-covariance matrix
                cross += covs[i] @ A @ covs[j]
            # Assign the cross-covariance matrix to the corresponding blocks in the joint covariance matrix
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
        # Generate a list of covariance matrices and true features for each view using list comprehensions
        covs = [
            self._generate_covariance_matrix(view_features, structure)
            for view_features, structure in zip(self.view_features, self.structure)
        ]

        true_features = [
            self._generate_true_feature(cov, sparsity, view_positive)
            for cov, sparsity, view_positive, view_features in zip(
                covs, self.view_sparsity, self.positive, self.view_features
            )
        ]
        return covs, true_features

    def _generate_true_feature(self, cov, sparsity, view_positive):
        # Generate a true feature matrix for a single view using its covariance matrix and sparsity level

        # Generate a random weight matrix with normal distribution and the same shape as the covariance matrix
        weights = self._generate_weights_matrix(cov.shape[0])

        # Apply a sparsity mask to the weight matrix to make some elements zero
        weights = self._apply_sparsity_mask(weights, sparsity)

        # Make the weight matrix positive if needed by taking the absolute value of the elements
        if view_positive:
            weights = self._make_weights_positive(weights)

        # Decorrelate the weight matrix from the covariance matrix and normalize it
        weights = _decorrelate_dims(weights, cov)
        weights /= np.sqrt(np.diag((weights.T @ cov @ weights)))
        return weights

    def _generate_weights_matrix(self, view_features):
        # Generate a random matrix with normal distribution and the given number of rows and latent dimensions
        return self.random_state.randn(view_features, self.latent_dims)

    def _apply_sparsity_mask(self, weights, sparsity):
        view_features = weights.shape[0]
        # convert sparsity to an integer number of nonzero elements
        if sparsity <= 1:
            sparsity = np.ceil(sparsity * view_features).astype("int")
        # create a binary mask with sparsity number of ones
        mask = np.stack(
            (
                np.concatenate(
                    ([0] * (view_features - sparsity), [1] * sparsity)
                ).astype(bool),
            )
            * self.latent_dims,
            axis=0,
        ).T
        # shuffle the mask randomly
        mask = mask.flatten()
        self.random_state.shuffle(mask)
        mask = mask.reshape(weights.shape)
        # apply the mask to the weights matrix
        return weights * mask

    def _make_weights_positive(self, weights):
        # set all negative elements to zero
        return np.abs(weights)

    def sample(self, n: int):
        X = np.zeros((n, self.chol.shape[0]))
        for i in range(n):
            X[i, :] = self._chol_sample(self.chol)
        views = np.split(X, np.cumsum(self.view_features)[:-1], axis=1)
        return views

    def _chol_sample(self, chol):
        rng = check_random_state(self.random_state)
        return chol @ rng.randn(chol.shape[0])


def _decorrelate_dims(up, cov):
    A = up.T @ cov @ up
    for k in range(1, A.shape[0]):
        up[:, k:] -= np.outer(up[:, k - 1], A[k - 1, k:] / A[k - 1, k - 1])
        A = up.T @ cov @ up
    return up
