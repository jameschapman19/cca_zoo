"""
Module for finding _CCALoss effects sequentially by deflation.

Check if each effect is significant, and if so, remove it from the data and repeat.
"""
from abc import ABCMeta
from typing import Iterable

import numpy as np
from sklearn.base import MetaEstimatorMixin

from cca_zoo._base import _BaseModel
from cca_zoo.linear._iterative._deflation import deflate_views
from cca_zoo.model_selection._validation import permutation_test_score


class SequentialModel(MetaEstimatorMixin, _BaseModel, metaclass=ABCMeta):
    def __init__(
        self,
        estimator,
        estimator_hyperparams=None,
        permutation_test_params=None,
        latent_dimensions=None,  # Maximum number of latent dimensions to fit
        copy_data=True,
        accept_sparse=False,
        random_state=None,
        permutation_test=False,  # Whether to use permutation test to determine significance
        p_threshold=1e-3,  # Threshold for permutation test if used
        corr_threshold=0.0,  # Threshold for effect size if permutation test not used
    ):
        super().__init__(
            latent_dimensions=latent_dimensions,
            copy_data=copy_data,
            accept_sparse=accept_sparse,
            random_state=random_state,
        )
        # Check the estimator has 1 latent dimension or if it is GridSearchCV or RandomizedSearchCV that the base
        # estimator has 1 latent dimension
        if hasattr(estimator, "estimator"):
            if estimator.estimator.latent_dimensions != 1:
                raise ValueError(
                    "The estimator must have 1 latent dimension, but has {}".format(
                        estimator.estimator.latent_dimensions
                    )
                )
        elif estimator.latent_dimensions != 1:
            raise ValueError(
                "The estimator must have 1 latent dimension, but has {}".format(
                    estimator.latent_dimensions
                )
            )
        self.estimator = estimator
        if estimator_hyperparams is None:
            estimator_hyperparams = {}
        self.estimator_hyperparams = estimator_hyperparams
        self.permutation_test = permutation_test
        if permutation_test_params is None:
            permutation_test_params = {}
        self.permutation_test_params = permutation_test_params
        self.p_threshold = p_threshold
        self.corr_threshold = corr_threshold

    def fit(self, views: Iterable[np.ndarray], y=None, **kwargs):
        # Validate the input data and parameters
        self._validate_data(views)
        self._check_params()
        # Set the default latent dimensions to the minimum number of features
        if self.latent_dimensions is None:
            self.latent_dimensions = min([view.shape[1] for view in views])
        # Initialize the weights_ and p-values lists
        self.weights_ = [[] for view in views]
        self.p_values = []
        # Loop over the latent dimensions
        k = 0
        while k < self.latent_dimensions:
            # Fit the estimator with the current representations
            self.estimator.set_params(**self.estimator_hyperparams)
            self.estimator.fit(views)
            # Perform permutation test if required
            p_value = None
            best_estimator = self.estimator
            if self.permutation_test:
                # Get the best estimator if it exists, otherwise use the original estimator
                best_estimator = getattr(
                    self.estimator, "best_estimator_", self.estimator
                )
                # Get the p-value from the permutation test score
                p_value = permutation_test_score(
                    best_estimator,
                    views,
                    y=None,
                    **self.permutation_test_params,
                )[2]
                # Append the p-value to the list
                self.p_values.append(p_value)
            # Check if the stopping criterion is met based on p-value or correlation score
            if (
                p_value is not None and p_value >= self.p_threshold
            ) or best_estimator.score(views) < self.corr_threshold:
                if p_value is not None:
                    self.p_values.pop()
                break
            else:
                # Deflate the representations and store the weights_
                views = deflate_views(views, best_estimator.weights_)
                for i, weight in enumerate(best_estimator.weights_):
                    self.weights_[i].append(weight)
                k += 1

        # Safety check to ensure the loop hasn't resulted in empty weights
        if all(len(w) == 0 for w in self.weights_):
            raise ValueError("No significant latent dimensions found.")

        # Set the final latent dimensions to k
        self.latent_dimensions = k
        # Concatenate the weights_ from each effect
        self.weights_ = [np.concatenate(weights, axis=1) for weights in self.weights_]
        return self
