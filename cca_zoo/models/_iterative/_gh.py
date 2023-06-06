import torch

from cca_zoo.models._iterative._base import BaseIterative
from cca_zoo.models._iterative._ey import EYLoop


class CCAGH(BaseIterative):
    """
    A class used to fit Regularized CCA by GHA-GEP

    Parameters
    ----------
    latent_dims : int, optional
        Number of latent dimensions to use, by default 1
    copy_data : bool, optional
        Whether to copy the data, by default True
    random_state : int, optional
        Random state to use, by default None
    accept_sparse : bool, optional
        Whether to accept sparse data, by default None
    batch_size : int, optional
        Batch size to use, by default 1
    epochs : int, optional
        Number of epochs to use, by default 1
    learning_rate : float, optional
        Learning rate to use, by default 0.01
    c : float, optional
        Regularization parameter, by default 0

    References
    ----------
    Chapman, James, Ana Lawry Aguila, and Lennie Wells. "A Generalized EigenGame with Extensions to Multiview Representation Learning." arXiv preprint arXiv:2211.11323 (2022).
    """

    def __init__(
        self,
        latent_dims: int = 1,
        copy_data=True,
        random_state=None,
        tol=1e-9,
        accept_sparse=None,
        batch_size=None,
        epochs=1,
        learning_rate=1,
        initialization="random",
        optimizer_kwargs=None,
    ):
        super().__init__(
            latent_dims=latent_dims,
            copy_data=copy_data,
            random_state=random_state,
            tol=tol,
            accept_sparse=accept_sparse,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            initialization=initialization,
        )
        self.objective = "cca"
        self.optimizer_kwargs = optimizer_kwargs

    def _get_module(self, weights=None, k=None):
        return GHALoop(
            weights=weights,
            k=k,
            learning_rate=self.learning_rate,
            optimizer_kwargs=self.optimizer_kwargs,
            objective="cca",
        )

    def _more_tags(self):
        return {"multiview": True, "stochastic": True}


class PLSGHA(CCAGH):
    """
    A class used to fit PLS by GHA-GEP

    Parameters
    ----------
    latent_dims : int, optional
        Number of latent dimensions to use, by default 1
    copy_data : bool, optional
        Whether to copy the data, by default True
    random_state : int, optional
        Random state to use, by default None
    accept_sparse : bool, optional
        Whether to accept sparse data, by default None
    batch_size : int, optional
        Batch size to use, by default 1
    epochs : int, optional
        Number of epochs to use, by default 1
    learning_rate : float, optional
        Learning rate to use, by default 0.01

    References
    ----------
    Chapman, James, Ana Lawry Aguila, and Lennie Wells. "A Generalized EigenGame with Extensions to Multiview Representation Learning." arXiv preprint arXiv:2211.11323 (2022).
    """

    def __init__(
        self,
        latent_dims: int = 1,
        copy_data=True,
        random_state=None,
        tol=1e-9,
        accept_sparse=None,
        batch_size=1,
        epochs=1,
        learning_rate=1,
        initialization="random",
    ):
        super().__init__(
            latent_dims=latent_dims,
            copy_data=copy_data,
            accept_sparse=accept_sparse,
            random_state=random_state,
            tol=tol,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            initialization=initialization,
        )
        self.objective = "pls"

    def _get_module(self, weights=None, k=None):
        return GHALoop(
            weights=weights,
            k=k,
            learning_rate=self.learning_rate,
            optimizer_kwargs=self.optimizer_kwargs,
            objective="pls",
        )


class GHALoop(EYLoop):
    def get_AB(self, z):
        # Getting the dimensions of the encoded views
        N, D = z[0].size()
        # Computing the covariance matrix of the concatenated views
        C = torch.cov(torch.hstack(z).T)
        # Extracting the cross-covariance and auto-covariance matrices
        A = C[:D, D:] + C[D:, :D]
        if self.objective == "cca":
            B = C[:D, :D] + C[D:, D:]
            A+=B
        else: # Only CCA is implemented for now
            raise NotImplementedError
        return A, B

    def loss(self, views, views2=None, **kwargs):
        # Encoding the views with the forward method
        z = self(views)
        # Getting A and B matrices from z
        A, B = self.get_AB(z)
        if views2 is None:
            # Computing rewards and penalties using A and B only
            rewards = torch.trace(2 * A)
            penalties = torch.trace(A @ B)
        else:
            # Encoding another set of views with the forward method
            z2 = self(views2)
            # Getting A' and B' matrices from z2
            _, B_ = self.get_AB(z2)
            # Computing rewards and penalties using A and B'
            rewards = torch.trace(2 * A)
            penalties = torch.trace(A @ B_)
        return {
            "objective": -rewards + penalties,
            "rewards": rewards,
            "penalties": penalties,
        }
