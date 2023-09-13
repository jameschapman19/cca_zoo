import itertools
from abc import abstractmethod
from typing import Iterable, Union, Any

import numpy as np
from cca_zoo.linear._pls import MPLS
from tqdm import tqdm

from cca_zoo._base import BaseModel
from cca_zoo.linear._dummy import DummyCCA, DummyPLS
from cca_zoo.linear._mcca import MCCA


class BaseIterative(BaseModel):
    def __init__(
        self,
        latent_dimensions: int = 1,
        copy_data=True,
        random_state=None,
        tol=1e-3,
        accept_sparse=None,
        epochs=100,
        initialization: Union[str, callable] = "random",
        early_stopping=False,
        verbose=True,
    ):
        super().__init__(
            latent_dimensions=latent_dimensions,
            copy_data=copy_data,
            random_state=random_state,
            accept_sparse=accept_sparse,
        )
        self.tol = tol
        self.epochs = epochs
        # validate the initialization method
        if initialization not in ["random", "uniform", "unregularized", "pls"]:
            raise ValueError(
                "Initialization method must be one of ['random', 'uniform', 'unregularized', 'pls']"
            )
        else:
            self.initialization = initialization
        # validate the callbacks
        self.verbose = verbose
        # validate the convergence checking
        self.early_stopping = early_stopping

    def fit(self, views: Iterable[np.ndarray], y=None, **kwargs):
        self.weights = self._fit(views)
        return self

    def _fit(self, views: Iterable[np.ndarray]):
        views = self._validate_data(views)
        self._initialize(views)
        self._check_params()
        # Solve using alternating optimisation across the views until convergence
        # Initialize the loss and the previous weights
        loss = np.inf
        prev_weights = self.weights.copy()
        # Loop over the epochs
        for epoch in tqdm(
            range(self.epochs),
            desc="Epochs",
            position=0,
            leave=True,
            disable=not self.verbose,
        ):
            # Loop over the views
            for i in range(len(views)):
                # Update the weights for the current view by solving a linear system
                self.weights[i] = self._update_weights(views, i)
            # Check if the loss has decreased enough
            curr_loss = self._objective(views)
            if self.early_stopping:
                weight_diff = np.sum(
                    [
                        np.linalg.norm(w - pw)
                        for w, pw in zip(self.weights, prev_weights)
                    ]
                ) / len(self.weights)
                if weight_diff < self.tol:
                    print(f"Early stopping at epoch {epoch}")
                    break
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
        for x, y in itertools.product(transformed_views, repeat=2):
            all_covs.append(
                np.diag(
                    np.corrcoef(x.T, y.T)[
                        : self.latent_dimensions, self.latent_dimensions :
                    ]
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


def _default_initializer(
    initialization: str, random_state: Any, latent_dims: int, pls: bool
) -> Union[DummyCCA, DummyPLS, MPLS, MCCA]:
    initializer_map = {
        "random": (DummyPLS if pls else DummyCCA)(
            latent_dims, random_state=random_state, uniform=False
        ),
        "uniform": (DummyPLS if pls else DummyCCA)(
            latent_dims, random_state=random_state, uniform=True
        ),
        "unregularized": (MPLS if pls else MCCA)(
            latent_dims, random_state=random_state
        ),
        "pls": MPLS(
            latent_dims, random_state=random_state
        ),  # Assuming PLS initialization can be used for both pls=True/False
    }

    initializer = initializer_map.get(initialization)

    if not initializer:
        raise ValueError(
            f"Initialization {initialization} not supported. Pass a generator implementing this method"
        )

    return initializer
