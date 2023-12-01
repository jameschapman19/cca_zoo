import itertools
from abc import abstractmethod
from typing import Iterable, Union, Any

import numpy as np
from sklearn.utils import check_random_state
from tqdm import tqdm

from cca_zoo._base import _BaseModel
from cca_zoo.linear._dummy import DummyCCA, DummyPLS
from cca_zoo.linear._mcca import MCCA
from cca_zoo.linear._pls import MPLS


class _BaseIterative(_BaseModel):
    def __init__(
        self,
        latent_dimensions: int = 1,
        copy_data=True,
        random_state=None,
        tol=1e-6,
        accept_sparse=None,
        epochs=500,
        initialization: Union[str, callable] = "uniform",
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
        self.weights_ = self._fit(views)
        return self

    def _fit(self, views: Iterable[np.ndarray]):
        views = self._validate_data(views)
        self.random_state = check_random_state(self.random_state)
        self._initialize(views)
        self._check_params()
        # Solve using alternating optimisation across the representations until convergence
        prev_weights = self.weights_.copy()
        # Loop over the epochs
        for epoch in tqdm(
            range(self.epochs),
            desc="Epochs",
            position=0,
            leave=True,
            disable=not self.verbose,
        ):
            # Loop over the representations
            for i in range(len(views)):
                # Update the weights_ for the current view by solving a linear system
                self.weights_[i] = self._update_weights(views, i)
            if all(
                [
                    np.linalg.norm(self.weights_[j] - prev_weights[j]) < self.tol
                    for j in range(len(views))
                ]
            ):
                break
            prev_weights = self.weights_.copy()
        # Return the final weights_
        return self.weights_

    @abstractmethod
    def _update_weights(self, view: np.ndarray, i: int):
        """Update the CCA weights_ for a given view.

        Parameters
        ----------
        view : np.ndarray
            The input view to update the CCA weights_ for
        i : int
            The index of the view

        Returns
        -------
        np.ndarray
            The updated CCA weights_ for the view
        """
        pass

    def _objective(self, views: Iterable[np.ndarray]):
        # Compute the objective function value for a given set of representations using SCCA
        # Get the scores of all representations
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
        """Initialize the CCA weights_ using the initialization method or function.

        Parameters
        ----------
        views : Iterable[np.ndarray]
            The input representations to initialize the CCA weights_ from
        """
        pls = self._get_tags().get("pls", False)
        initializer = _default_initializer(
            self.initialization, self.random_state, self.latent_dimensions, pls
        )
        # Fit the initializer on the input representations and get the weights_ as numpy arrays
        self.weights_ = initializer.fit(views).weights_
        self.weights_ = [weights.astype(np.float32) for weights in self.weights_]

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
