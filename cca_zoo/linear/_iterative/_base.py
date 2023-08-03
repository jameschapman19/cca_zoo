import itertools
from abc import abstractmethod
from typing import Iterable, Union

import numpy as np

from cca_zoo._base import BaseModel
from cca_zoo.linear._dummy import DummyCCA, DummyPLS
from cca_zoo.linear._mcca import MCCA
from cca_zoo.linear._pls import MPLS


class BaseIterative(BaseModel):
    def __init__(
        self,
        latent_dimensions: int = 1,
        copy_data=True,
        random_state=None,
        tol=1e-3,
        accept_sparse=None,
        epochs=100,
        val_split=None,
        learning_rate=1,
        initialization: Union[str, callable] = "random",
        early_stopping=False,
        patience=10,
        verbose=None,
    ):
        super().__init__(
            latent_dimensions=latent_dimensions,
            copy_data=copy_data,
            random_state=random_state,
            accept_sparse=accept_sparse,
        )
        self.tol = tol
        self.epochs = epochs
        # validate the split
        if val_split is not None:
            if val_split <= 0 or val_split >= 1:
                raise ValueError("Validation split must be between 0 and 1")
        self.val_split = val_split
        self.learning_rate = learning_rate
        # validate the initialization method
        if initialization not in ["random", "uniform", "unregularized", "pls"]:
            raise ValueError(
                "Initialization method must be one of ['random', 'uniform', 'unregularized', 'pls']"
            )
        else:
            self.initialization = initialization
        # validate the callbacks
        self.verbose = verbose
        self.patience = patience
        # validate the convergence checking
        self.early_stopping = early_stopping

    def fit(self, views: Iterable[np.ndarray], y=None, **kwargs):
        self.weights = self._fit(views)
        return self

    def _fit(self, views: Iterable[np.ndarray]):
        self._validate_data(views)
        self._initialize(views)
        self._check_params()
        # Solve using alternating optimisation across the views until convergence
        # Initialize the loss and the previous weights
        loss = np.inf
        prev_weights = self.weights.copy()
        # Loop over the epochs
        for epoch in range(self.epochs):
            # Loop over the views
            for i in range(len(views)):
                # Update the weights for the current view by solving a linear system
                self.weights[i] = self._update_weights(views, i)
            # Compute the current loss using the objective function
            curr_loss = self._objective(views)
            # Check if the loss has decreased enough
            if np.abs(curr_loss - loss) < self.tol and self.early_stopping:
                # If yes, break the loop
                break
            else:
                # If not, update the loss and the previous weights
                loss = curr_loss
                prev_weights = self.weights.copy()
        # Return the final weights
        return self.weights

    @abstractmethod
    def _update_weights(self, view: np.ndarray, i: int):
        """Update the CCA weights for a given view.

        Parameters
        ----------
        view : np.ndarray
            The input view to update the CCA weights for
        i : int
            The index of the view

        Returns
        -------
        np.ndarray
            The updated CCA weights for the view
        """
        pass

    def _objective(self, views: Iterable[np.ndarray]):
        # Compute the objective function value for a given set of views using SCCA
        # Get the scores of all views
        transformed_views = self.transform(views)
        all_covs = []
        # Sum all the pairwise covariances except self covariance
        for i, j in itertools.combinations(range(len(transformed_views)), 2):
            if i != j:
                all_covs.append(
                    np.cov(
                        np.hstack(
                            (
                                transformed_views[i],
                                transformed_views[j],
                            )
                        ).T
                    )
                )
        # the sum of covariances
        return np.sum(all_covs)

    def _initialize(self, views: Iterable[np.ndarray]):
        """Initialize the CCA weights using the initialization method or function.

        Parameters
        ----------
        views : Iterable[np.ndarray]
            The input views to initialize the CCA weights from
        """
        pls = self._get_tags().get("pls", False)
        initializer = _default_initializer(
            self.initialization, self.random_state, self.latent_dimensions, pls
        )
        # Fit the initializer on the input views and get the weights as numpy arrays
        self.weights = initializer.fit(views).weights
        self.weights = [weights.astype(np.float32) for weights in self.weights]

    def _more_tags(self):
        # Indicate that this class is for multiview data
        return {"iterative": True}


def _default_initializer(initialization, random_state, latent_dims, pls):
    if pls:
        if initialization == "random":
            initializer = DummyPLS(
                latent_dims, random_state=random_state, uniform=False
            )
        elif initialization == "uniform":
            initializer = DummyPLS(latent_dims, random_state=random_state, uniform=True)
        elif initialization == "unregularized":
            initializer = MPLS(latent_dims, random_state=random_state)
        else:
            raise ValueError(
                "Initialization {type} not supported. Pass a generator implementing this method"
            )
    else:
        if initialization == "random":
            initializer = DummyCCA(
                latent_dims, random_state=random_state, uniform=False
            )
        elif initialization == "uniform":
            initializer = DummyCCA(latent_dims, random_state=random_state, uniform=True)
        elif initialization == "unregularized":
            initializer = MCCA(latent_dims, random_state=random_state)
        elif initialization == "pls":
            initializer = MPLS(latent_dims, random_state=random_state)
        else:
            raise ValueError(
                "Initialization {type} not supported. Pass a generator implementing this method"
            )
    return initializer
