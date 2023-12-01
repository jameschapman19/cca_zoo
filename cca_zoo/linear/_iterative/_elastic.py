from typing import Union, Iterable

import numpy as np
from sklearn.linear_model import ElasticNet, Lasso, Ridge, SGDRegressor

from cca_zoo._utils._checks import _process_parameter
from cca_zoo.linear._iterative._base import _BaseIterative
from cca_zoo.linear._iterative._deflation import _DeflationMixin


# Define SCCA_Elastic class
class ElasticCCA(_DeflationMixin, _BaseIterative):
    def __init__(
        self,
        latent_dimensions: int = 1,
        copy_data=True,
        random_state=None,
        tol=1e-3,
        accept_sparse=None,
        epochs=100,
        initialization: Union[str, callable] = "uniform",
        early_stopping=False,
        verbose=None,
        alpha=None,  # regularization parameter for Elastic
        l1_ratio=None,  # ratio of L1 to L2 penalty for Elastic
        positive=None,  # whether to enforce positive coefficients for Elastic
        stochastic=False,  # whether to use stochastic gradient descent for Elastic
    ):
        super().__init__(
            latent_dimensions=latent_dimensions,
            copy_data=copy_data,
            random_state=random_state,
            tol=tol,
            accept_sparse=accept_sparse,
            epochs=epochs,
            initialization=initialization,
            early_stopping=early_stopping,
            verbose=verbose,
        )
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.positive = positive
        self.stochastic = stochastic

    def _check_params(self):
        self.alpha = _process_parameter("alpha", self.alpha, 0, self.n_views_)
        self.l1_ratio = _process_parameter("l1_ratio", self.l1_ratio, 0, self.n_views_)
        self.positive = _process_parameter(
            "positive", self.positive, False, self.n_views_
        )
        self.regressors = initialize_regressors(
            self.alpha,
            self.l1_ratio,
            self.positive,
            self.stochastic,
            self.tol,
            self.random_state,
        )

    def _update_weights(self, views: Iterable[np.ndarray], i: int):
        # Update the weights_ for the current view using Elastic
        # Get the scores of all representations
        scores = np.stack(self.transform(views))
        # Compute the target by summing the scores along dim 0 and dividing by the square root of the covariance of
        # the target
        target = np.sum(scores, axis=0)
        target = target / np.linalg.norm(target)

        # Loop over the representations and fit each regressor to the view and the target
        self.regressors[i] = self.regressors[i].fit(views[i], target)
        # Update the weights_ with the coefficients of each regressor
        new_weights = np.atleast_2d(self.regressors[i].coef_).T
        # Return the updated weights_
        return new_weights

    def _objective(self, views: Iterable[np.ndarray]):
        scores = np.stack(self.transform(views))
        objective = 0
        for view_index, view in enumerate(views):
            # create a mask that is True for elements not equal to k along dim k
            mask = np.arange(scores.shape[0]) != view_index
            # apply the mask to scores and sum along dim k
            target = np.sum(scores[mask], axis=0)
            objective += elastic_objective(
                scores[view_index],
                target,
                self.regressors[view_index].coef_,
                self.alpha[view_index],
                self.l1_ratio[view_index],
            )
        return objective


# Define SCCA_IPLS class
class SCCA_IPLS(_DeflationMixin, _BaseIterative):
    def __init__(
        self,
        latent_dimensions: int = 1,
        copy_data=True,
        random_state=None,
        tol=1e-3,
        accept_sparse=None,
        epochs=100,
        initialization: Union[str, callable] = "uniform",
        early_stopping=False,
        verbose=True,
        alpha=None,  # regularization parameter for Elastic
        l1_ratio=1,  # ratio of L1 to L2 penalty for Elastic
        positive=None,  # whether to enforce positive coefficients for Elastic
        stochastic=False,  # whether to use stochastic gradient descent for Elastic
    ):
        super().__init__(
            latent_dimensions=latent_dimensions,
            copy_data=copy_data,
            random_state=random_state,
            tol=tol,
            accept_sparse=accept_sparse,
            epochs=epochs,
            initialization=initialization,
            early_stopping=early_stopping,
            verbose=verbose,
        )
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.positive = positive
        self.stochastic = stochastic

    def _check_params(self):
        self.alpha = _process_parameter("alpha", self.alpha, 0, self.n_views_)
        self.positive = _process_parameter(
            "positive", self.positive, False, self.n_views_
        )
        self.l1_ratio = _process_parameter("l1_ratio", self.l1_ratio, 1, self.n_views_)
        self.regressors = initialize_regressors(
            self.alpha,
            self.l1_ratio,
            self.positive,
            self.stochastic,
            self.tol,
            self.random_state,
        )

    def _update_weights(self, views: Iterable[np.ndarray], i: int):
        # Update the weights_ for the current view using IPLS
        # Get the scores of all representations
        scores = np.stack(self.transform(views))

        # Create a mask that is True for elements not equal to j along dim j
        mask = np.arange(scores.shape[0]) != i
        # Apply the mask to scores and sum along dim j
        target = np.sum(scores[mask], axis=0)
        # Solve the elastic regression
        self.regressors[i] = self.regressors[i].fit(views[i], target)
        # Normalize the coefficients by dividing by the square root of the covariance of the view and the coefficients
        self.regressors[i].coef_ /= np.sqrt(np.cov(views[i] @ self.regressors[i].coef_))
        # Update the weights_ with the coefficients of each regressor
        new_weights = self.regressors[i].coef_
        # Return the updated weights_
        return new_weights[:, None]

    def _objective(self, views: Iterable[np.ndarray]):
        scores = np.stack(self.transform(views))
        objective = 0
        for view_index, view in enumerate(views):
            # create a mask that is True for elements not equal to k along dim k
            mask = np.arange(scores.shape[0]) != view_index
            # apply the mask to scores and sum along dim k
            target = np.sum(scores[mask], axis=0)
            objective += elastic_objective(
                scores[view_index],
                target,
                self.regressors[view_index].coef_,
                self.alpha[view_index],
                self.l1_ratio[view_index],
            )
        return objective


def initialize_regressors(alpha, l1_ratio, positive, stochastic, tol, random_state):
    regressors = []
    for alpha, l1_ratio, positive in zip(alpha, l1_ratio, positive):
        if stochastic:
            regressors.append(
                SGDRegressor(
                    penalty="elasticnet",
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    fit_intercept=False,
                    tol=tol,
                    warm_start=True,
                    random_state=random_state,
                )
            )
        elif l1_ratio == 0:
            regressors.append(
                Ridge(
                    alpha=alpha,
                    fit_intercept=False,
                    positive=positive,
                    random_state=random_state,
                    tol=tol,
                )
            )
        elif l1_ratio == 1:
            regressors.append(
                Lasso(
                    alpha=alpha,
                    fit_intercept=False,
                    warm_start=True,
                    positive=positive,
                    random_state=random_state,
                    tol=tol,
                    selection="random",
                )
            )
        else:
            regressors.append(
                ElasticNet(
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    fit_intercept=False,
                    warm_start=True,
                    positive=positive,
                    random_state=random_state,
                    tol=tol,
                    selection="random",
                )
            )
    return regressors


def elastic_objective(z, y, w, alpha, l1_ratio):
    n = len(y)
    objective = np.linalg.norm(z - y) ** 2 / (2 * n)
    l1_pen = alpha * l1_ratio * np.linalg.norm(w, ord=1)
    l2_pen = alpha * (1 - l1_ratio) * np.linalg.norm(w, ord=2)
    return objective + l1_pen + l2_pen
