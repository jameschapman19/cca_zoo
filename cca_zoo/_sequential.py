"""
Module for finding CCA effects sequentially by deflation.

Check if each effect is significant, and if so, remove it from the data and repeat.
"""
from abc import ABCMeta

from sklearn.base import MetaEstimatorMixin
from cca_zoo._base import BaseModel
from cca_zoo.model_selection._validation import permutation_test_score


class SequentialModel(MetaEstimatorMixin, BaseModel, metaclass=ABCMeta):
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
        # Check the estimator has 1 latent dimension or if it is GridSearchCV or RandomizedSearchCV that the base estimator has 1 latent dimension
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

    def fit(self, views):
        """
        Fits the model to the views.

        Parameters
        ----------
        views : list of array-likes of shape (n_samples, n_features)
            A list of views, where each view is an array-like matrix of shape (n_samples, n_features).

        Returns
        -------
        self : SequentialModel
            The SequentialModel instance.
        """
        # Validate the input data and parameters
        self._validate_data(views)
        self._check_params()
        # Set the default latent dimensions to the minimum number of features
        if self.latent_dimensions is None:
            self.latent_dimensions = min([view.shape[1] for view in views])
        # Initialize the weights and p-values lists
        self.weights = [[] for view in views]
        self.p_values = []
        # Loop over the latent dimensions
        for k in range(self.latent_dimensions):
            # Fit the estimator with the current views
            self.estimator.set_params(**self.estimator_hyperparams)
            self.estimator.fit(views)
            # Perform permutation test if required
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
                p_value >= self.p_threshold
                or best_estimator.score(views) < self.corr_threshold
            ):
                break
            else:
                # Deflate the views and store the weights
                views = self.deflate(views)
                for i, weight in enumerate(best_estimator.weights):
                    self.weights[i].append(weight)
        # Set the final latent dimensions to k
        self.latent_dimensions = k
        # Concatenate the weights from each effect
        self.weights = [np.concatenate(weights, axis=1) for weights in self.weights]
        return self

    def deflate(self, views):
        # deflate by projection deflation
        scores = self.estimator.transform(views)
        views = [
            view - np.outer(score, score) @ view / np.dot(score.T, score)
            for view, score in zip(views, scores)
        ]
        return views


if __name__ == "__main__":
    import numpy as np

    from cca_zoo.linear import rCCA
    from cca_zoo.data.simulated import LinearSimulatedData
    from cca_zoo.model_selection import GridSearchCV

    np.random.seed(0)
    data = LinearSimulatedData(view_features=[10, 10], latent_dims=5, correlation=0.8)
    X, Y = data.sample(200)

    rcca = rCCA()
    # pipe=Pipeline([
    #     ('preprocessing', MultiViewPreprocessing((StandardScaler(), StandardScaler()))),
    #     ('rcca', rcca)
    # ])
    # pipe=Pipeline([('rcca', rcca)])
    pipe = rcca
    param_grid = {
        "c": [0.1, 0.2, 0.3],
    }
    gs = GridSearchCV(pipe, param_grid, cv=2, verbose=1, n_jobs=1)

    model = SequentialModel(
        gs, latent_dimensions=10, permutation_test=True, p_threshold=0.05
    )

    model.fit([X, Y])

    print(model.weights)
