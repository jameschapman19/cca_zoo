from typing import Union

import torch

from cca_zoo.models._base import PLSMixin
from cca_zoo.models._iterative._base import BaseIterative
from cca_zoo.models._iterative._ey import EYLoop


class CCASVD(BaseIterative):
    """
    A class used to fit Regularized CCA by Delta-EigenGame

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
        batch_size=None,
        epochs=1,
        learning_rate=1e-1,
        initialization: Union[str, callable] = "random",
        line_search=False,
        rho=0.1,
        ensure_descent=False,
        dataloader_kwargs=None,
        optimizer_kwargs=None,
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
            dataloader_kwargs=dataloader_kwargs,
        )
        self.rho = rho
        self.line_search = line_search
        self.ensure_descent = ensure_descent
        self.velocity = None
        self.previous_views = None
        self.optimizer_kwargs = optimizer_kwargs

    def _get_module(self, weights=None, k=None):
        return SVDLoop(
            weights=weights,
            k=k,
            learning_rate=self.learning_rate,
            optimizer_kwargs=self.optimizer_kwargs,
            objective="cca",
        )

    def _more_tags(self):
        return {"multiview": True, "stochastic": True}


class PLSSVD(CCASVD, PLSMixin):
    """
    A class used to fit Regularized CCA by Delta-EigenGame

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
        batch_size=None,
        epochs=1,
        learning_rate=1,
        initialization: Union[str, callable] = "random",
        line_search=False,
        rho=0.1,
        ensure_descent=False,
        dataloader_kwargs=None,
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
            dataloader_kwargs=dataloader_kwargs,
        )
        self.rho = rho
        self.line_search = line_search
        self.ensure_descent = ensure_descent
        self.velocity = None
        self.previous_views = None

    def _get_module(self, weights=None, k=None):
        return SVDLoop(
            weights=weights,
            k=k,
            learning_rate=self.learning_rate,
            optimizer_kwargs=self.optimizer_kwargs,
            objective="pls",
        )

    def _more_tags(self):
        return {"multiview": True, "stochastic": True}


class SVDLoop(EYLoop):
    def loss(self, views, views2=None, **kwargs):
        z = self(views)
        C = torch.cov(torch.hstack(z).T)
        latent_dims = z[0].shape[1]

        Cxy = C[:latent_dims, latent_dims:]
        Cxx = C[:latent_dims, :latent_dims]

        if views2 is None:
            Cyy = C[latent_dims:, latent_dims:]
        else:
            z2 = self(views2)
            Cyy = torch.cov(torch.hstack(z2).T)[latent_dims:, latent_dims:]

        rewards = torch.trace(2 * Cxy)
        penalties = torch.trace(Cxx @ Cyy)

        return {
            "objective": -rewards + penalties,
            "rewards": rewards,
            "penalties": penalties,
        }
