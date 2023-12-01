import itertools
from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np
import scipy
from scipy.sparse import block_diag
from sklearn.datasets import make_spd_matrix
from sklearn.utils.validation import check_random_state

from cca_zoo._utils._checks import _process_parameter


def cov_eigvals(X):
    S = np.linalg.svd(X, compute_uv=False) ** 2
    return S.sum()


class _BaseData(ABC):
    def __init__(
        self,
        view_features: List[int],
        latent_dimensions: int = 1,
        random_state: Union[int, np.random.RandomState] = None,
        rank: int = None,
        density: float = 1.0,
    ):
        self.view_features = view_features
        self.latent_dimensions = latent_dimensions
        self.random_state = check_random_state(random_state)
        self.rank = min(view_features) if rank is None else rank
        self.density = density

    def _generate_covariance_matrix(self, features: int, structure: str):
        """
        Generates a covariance matrix for a view based on the specified structure.

        :param features: Number of features in the view.
        :param structure: Structure type of the covariance matrix.
        :return: A covariance matrix for the view.
        """
        if structure == "identity":
            return np.eye(features)
        else:
            covariance_matrix = make_spd_matrix(
                features, random_state=self.random_state
            )
            return self._scale_covariance_matrix(covariance_matrix)

    @staticmethod
    def _scale_covariance_matrix(covariance_matrix):
        """
        Scales a covariance matrix so that its diagonal elements are 1, effectively
        converting it to a correlation matrix.

        :param cov_matrix: A numpy array representing the covariance matrix.
        :return: Scaled covariance matrix with 1s on the diagonal.
        """
        # Extract the standard deviations from the diagonal of the covariance matrix
        std_devs = np.sqrt(np.diag(covariance_matrix))

        # Scale the covariance matrix
        scaled_cov_matrix = covariance_matrix / np.outer(std_devs, std_devs)

        return scaled_cov_matrix

    def _covariance_factor(self, features, structure):
        """
        Generates a covariance factor matrix based on the specified structure. For a large number of
        features, it generates a sparse representation to reduce memory and computational demands.

        :param features: Number of features in the view.
        :param structure: Structure type of the covariance matrix.
        :return: A covariance factor matrix for the view.
        """
        if structure == "identity":
            factor = scipy.sparse.eye(features, features)
            factor /= np.sqrt(features)
            return factor
        else:
            factor = scipy.sparse.random(
                features,
                self.rank,
                density=self.density,
                random_state=self.random_state,
            )
            s = scipy.linalg.svdvals(factor.toarray())
            factor /= np.sqrt(s.sum())
            return factor

    def covariance_matrices(self):
        return [factor @ factor.T for factor in self.covariance_factors]

    @abstractmethod
    def sample(self, n_samples: int):
        pass


class LatentVariableData(_BaseData):
    """
    This class generates data based on latent variable models. It allows for the specification
    of various parameters including the sparsity and structure of the data views, and the
    signal-to-noise ratio. It also supports sparse covariance matrix factorization to
    handle scenarios with a high number of features efficiently, reducing memory and
    computational demands.
    """

    def __init__(
        self,
        view_features: List[int],
        latent_dimensions: int = 1,
        random_state: Union[int, np.random.RandomState] = None,
        sparsity_levels: Union[List[float], float] = None,
        positivity_constraints: Union[bool, List[bool]] = False,
        covariance_structure: str = "identity",
        signal_to_noise_ratio: float = 1.0,
        rank: int = None,
        density: float = 1.0,
    ):
        """
        Initializes the LatentVariableData class with specified parameters.

        :param view_features: List of integers representing the number of features in each view.
        :param latent_dimensions: The number of latent dimensions in the data.
        :param random_state: The random state for reproducibility.
        :param sparsity_levels: A float or a list of floats specifying the sparsity level for each view.
        :param positivity_constraints: A boolean or a list of booleans indicating if the loadings should be positive.
        :param covariance_structure: Specifies the structure of the covariance matrix ('identity' or other).
        :param signal_to_noise_ratio: The signal-to-noise ratio in the data.
        :param rank: Maximum rank for sparse covariance matrix factorization, used for large feature sets.
        :param density: Density of the sparse matrix in sparse covariance matrix factorization.
        """
        super().__init__(view_features, latent_dimensions, random_state, rank, density)
        self.signal_to_noise_ratio = signal_to_noise_ratio
        self.sparsity_levels = _process_parameter(
            "sparsity_levels", sparsity_levels, 1.0, len(view_features)
        )
        self.positivity_constraints = _process_parameter(
            "positivity_constraints", positivity_constraints, False, len(view_features)
        )
        self.covariance_structure = _process_parameter(
            "covariance_structure", covariance_structure, "identity", len(view_features)
        )
        self.true_loadings = [
            self._generate_loading_matrix(features, sparsity, positivity)
            for features, sparsity, positivity in zip(
                self.view_features, self.sparsity_levels, self.positivity_constraints
            )
        ]
        self.covariance_factors = [
            self._covariance_factor(features, structure)
            for features, structure in zip(
                self.view_features, self.covariance_structure
            )
        ]

    def _generate_loading_matrix(
        self, features: int, sparsity: float, positivity: bool
    ):
        """
        Generates a loading matrix for a view based on the specified sparsity and positivity.

        :param features: Number of features in the view.
        :param sparsity: Sparsity level for the view.
        :param positivity: Whether to enforce positivity constraints.
        :return: A loading matrix for the view.
        """
        loading_matrix = self.random_state.standard_normal(
            size=(features, self.latent_dimensions)
        )
        if sparsity <= 1:
            sparsity = int(np.ceil(sparsity * features))
        mask_elements = [0] * (features - sparsity) + [1] * sparsity
        mask = np.stack([mask_elements] * loading_matrix.shape[1]).T
        np.random.shuffle(mask)
        loading_matrix *= mask
        if positivity:
            loading_matrix = np.abs(loading_matrix)
        # divide by sum of eigenvalues to normalize
        loading_matrix /= np.sqrt(
            np.linalg.eigvalsh(loading_matrix.T @ loading_matrix).sum()
        )
        return loading_matrix

    def sample(self, num_samples: int, return_latent: bool = False):
        """
        Generates samples from the latent variable model.

        :param num_samples: Number of samples to generate.
        :param return_latent: Whether to return the latent variables along with the views.
        :return: Generated views and optionally the latent variables.
        """
        latent_variables = self.random_state.multivariate_normal(
            np.zeros(self.latent_dimensions),
            np.eye(self.latent_dimensions),
            num_samples,
        )

        views = [
            latent_variables @ loading.T
            + self.random_state.standard_normal(
                size=(num_samples, covariance_factor.shape[1])
            )
            @ covariance_factor.T
            / np.sqrt(self.signal_to_noise_ratio)
            for loading, covariance_factor in zip(
                self.true_loadings, self.covariance_factors
            )
        ]
        return (views, latent_variables) if return_latent else views

    def joint_covariance_matrix(self):
        """
        Computes the joint covariance matrix for all views.

        :return: The joint covariance matrix.
        """
        joint_cov = np.zeros((sum(self.view_features), sum(self.view_features)))
        joint_cov[: self.view_features[0], : self.view_features[0]] = (
            self.true_loadings[0] @ self.true_loadings[0].T
            + self.covariance_matrices[0]
        )
        joint_cov[self.view_features[0] :, self.view_features[0] :] = (
            self.true_loadings[1] @ self.true_loadings[1].T
            + self.covariance_matrices[1]
        )
        joint_cov[: self.view_features[0], self.view_features[0] :] = (
            self.true_loadings[0] @ self.true_loadings[1].T
        )
        joint_cov[self.view_features[0] :, : self.view_features[0]] = (
            self.true_loadings[1] @ self.true_loadings[0].T
        )
        return joint_cov

    def true_features(self):
        """
        Estimates the true features based on the loading matrices and covariance matrices.

        :return: List of estimated true features for each view.
        """
        return [
            np.linalg.inv(cov + loading.T @ loading) @ loading
            for cov, loading in zip(self.covariance_matrices, self.true_loadings)
        ]


class JointData(_BaseData):
    """
    Class for generating simulated data for a linear model with multiple representations.
    """

    def __init__(
        self,
        view_features: List[int],
        latent_dimensions: int = 1,
        sparsity_levels: Union[List[float], float] = None,
        correlation: Union[List[float], float] = 0.99,
        covariance_structure: str = "random",
        positive: Union[bool, List[bool]] = False,
        random_state: Union[int, np.random.RandomState] = None,
        rank: int = None,
        density: float = 1.0,
    ):
        super().__init__(view_features, latent_dimensions, random_state, rank, density)
        self.correlation = _process_parameter(
            "correlation", correlation, 0.99, self.latent_dimensions
        )
        self.sparsity_levels = _process_parameter(
            "sparsity_levels", sparsity_levels, 1.0, len(view_features)
        )
        self.covariance_structure = _process_parameter(
            "covariance_structure", covariance_structure, "random", len(view_features)
        )
        self.positive = _process_parameter(
            "positive", positive, False, len(view_features)
        )
        self.covariance_factors = [
            self._covariance_factor(features, structure)
            for features, structure in zip(
                self.view_features, self.covariance_structure
            )
        ]
        self.true_features = [
            self._generate_true_weight(view_features, sparsity_levels, is_positive, cov)
            for view_features, sparsity_levels, is_positive, cov in zip(
                self.view_features,
                self.sparsity_levels,
                self.positive,
                self.covariance_factors,
            )
        ]
        self.true_loadings = [
            covariance_factor @ (covariance_factor.T @ weight)
            for weight, covariance_factor in zip(
                self.true_features, self.covariance_factors
            )
        ]
        U, S, Vt = np.linalg.svd(
            self._generate_joint_covariance(self.covariance_factors), full_matrices=True
        )
        self.US = U * np.sqrt(S)

    def _generate_true_weight(
        self, view_features, sparsity_levels, is_positive, covariance_factor
    ):
        loadings = self.random_state.randn(view_features, self.latent_dimensions)
        if sparsity_levels <= 1:
            sparsity_levels = np.ceil(sparsity_levels * loadings.shape[0]).astype(int)
        mask_elements = [0] * (loadings.shape[0] - sparsity_levels) + [
            1
        ] * sparsity_levels
        mask = np.stack([mask_elements] * loadings.shape[1]).T
        self.random_state.shuffle(mask)
        loadings *= mask
        if is_positive:
            loadings = np.abs(loadings)
        loadings = self._decorrelate_weights(loadings, covariance_factor)
        return loadings / np.sqrt(
            np.diag(loadings.T @ covariance_factor @ covariance_factor.T @ loadings)
        )

    def _generate_joint_covariance(self, covariance_factors):
        """Generates a joint covariance matrix for all representations."""
        joint_covariance = block_diag(
            [
                covariance_factor @ covariance_factor.T
                for covariance_factor in covariance_factors
            ]
        ).toarray()
        split_points = np.concatenate(([0], np.cumsum(self.view_features)))

        for i, j in itertools.combinations(range(len(split_points) - 1), 2):
            cross_cov = self._compute_cross_covariance(covariance_factors, i, j)
            joint_covariance[
                split_points[i] : split_points[i + 1],
                split_points[j] : split_points[j + 1],
            ] = cross_cov
            joint_covariance[
                split_points[j] : split_points[j + 1],
                split_points[i] : split_points[i + 1],
            ] = cross_cov.T
        return joint_covariance

    def _compute_cross_covariance(self, cov_factors, i, j):
        """Computes the cross-covariance matrix for a pair of representations."""
        cross_cov = np.zeros((self.view_features[i], self.view_features[j]))

        for _ in range(self.latent_dimensions):
            outer_product = np.outer(
                self.true_features[i][:, _], self.true_features[j][:, _]
            )
            cross_cov += (
                cov_factors[i]
                @ cov_factors[i].T
                @ (self.correlation[_] * outer_product)
                @ cov_factors[j]
                @ cov_factors[j].T
            )

        return cross_cov

    @staticmethod
    def _decorrelate_weights(weights, covariance_factor):
        product = (weights.T @ covariance_factor) @ (covariance_factor.T @ weights)
        for k in range(1, product.shape[0]):
            weights[:, k:] -= np.outer(
                weights[:, k - 1], product[k - 1, k:] / product[k - 1, k - 1]
            )
            product = (weights.T @ covariance_factor) @ (covariance_factor.T @ weights)
        return weights

    def sample(self, n_samples: int):
        random_data = self.random_state.standard_normal(
            size=(n_samples, self.US.shape[0])
        )
        samples = random_data @ self.US.T
        return np.split(samples, np.cumsum(self.view_features)[:-1], axis=1)
