from typing import Union

import numpy as np
import torch

from cca_zoo.models._plsmixin import PLSMixin
from cca_zoo.models._iterative._base import BaseGradientLoop, BaseIterative


class CCAEY(BaseIterative):
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
        epochs=100,
        learning_rate=1e-1,
        initialization: Union[str, callable] = "random",
        dataloader_kwargs=None,
        optimizer_kwargs=None,
        convergence_checking=False,
        patience=10,
        track=False,
        verbose=False,
    ):
        self.optimizer_kwargs = optimizer_kwargs
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
            convergence_checking=convergence_checking,
            patience=patience,
            track=track,
            verbose=verbose,
        )
        # ensure dataloader_kwargs['shuffle'] is True
        self.dataloader_kwargs["shuffle"] = True

    def _get_module(self, weights=None, k=None):
        return EYLoop(
            weights=weights,
            k=k,
            learning_rate=self.learning_rate,
            optimizer_kwargs=self.optimizer_kwargs,
            objective="cca",
        )

    def _more_tags(self):
        return {"multiview": True, "stochastic": True}


class PLSEY(CCAEY, PLSMixin):
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
        dataloader_kwargs=None,
        convergence_checking=False,
        patience=10,
        track=False,
        verbose=False,
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
            convergence_checking=convergence_checking,
            patience=patience,
            track=track,
            verbose=verbose,
        )
        self.previous_views = None

    def _get_module(self, weights=None, k=None):
        return EYLoop(
            weights=weights,
            k=k,
            learning_rate=self.learning_rate,
            optimizer_kwargs=self.optimizer_kwargs,
            objective="pls",
        )

    def _more_tags(self):
        return {"multiview": True, "stochastic": True}


class EYLoop(BaseGradientLoop):
    def __init__(
        self,
        weights=None,
        k=None,
        learning_rate=1e-3,
        optimizer_kwargs=None,
        objective="cca",
    ):
        super().__init__(
            weights=weights,
            k=k,
            learning_rate=learning_rate,
            optimizer_kwargs=optimizer_kwargs,
        )
        self.objective = objective
        self.batch_queue = []
        self.val_batch_queue = []

    def training_step(self, batch, batch_idx):
        # Checking if the queue has at least one batch
        if len(self.batch_queue) < 1:
            # Adding the current batch to the queue
            self.batch_queue.append(batch)
            # Returning a zero loss
            loss = {
                "loss": torch.tensor(0, requires_grad=True, dtype=torch.float32),
            }
        else:
            # randomly select a batch from the queue
            batch2 = self.batch_queue[np.random.randint(0, len(self.batch_queue))]
            # Computing the loss with the current batch and the oldest batch in the queue
            loss = self.loss(batch["views"], batch2["views"])
            # Adding the current batch to the queue and removing the oldest batch
            self.batch_queue.append(batch)
            self.batch_queue.pop(0)
        # Logging the loss components
        for k, v in loss.items():
            self.log(k, v, prog_bar=False)
        return loss

    def get_AB(self, z):
        latent_dims = z[0].shape[1]
        A = torch.zeros(
            latent_dims, latent_dims, device=z[0].device
        )  # initialize the cross-covariance matrix
        B = torch.zeros(
            latent_dims, latent_dims, device=z[0].device
        )  # initialize the auto-covariance matrix
        for i, zi in enumerate(z):
            for j, zj in enumerate(z):
                if i == j:
                    if self.objective == "cca":
                        B += torch.cov(
                            zi.T
                        )  # add the auto-covariance of each view to B
                    elif self.objective == "pls":
                        B += self.weights[i].T @ self.weights[i]
                else:
                    A += torch.cov(torch.hstack((zi, zj)).T)[
                        latent_dims:, :latent_dims
                    ]  # add the cross-covariance of each pair of views to A
        return A / len(z), B / len(
            z
        )  # return the normalized matrices (divided by the number of views)

    def loss(self, views, views2=None, **kwargs):
        # Encoding the views with the forward method
        z = self(views)
        # Getting A and B matrices from z
        A, B = self.get_AB(z)

        if views2 is None:
            # Computing rewards and penalties using A and B only
            rewards = torch.trace(2 * A)
            penalties = torch.trace(B @ B)

        else:
            # Encoding another set of views with the forward method
            z2 = self(views2)
            # Getting A' and B' matrices from z2
            A_, B_ = self.get_AB(z2)
            # Computing rewards and penalties using A and B'
            rewards = torch.trace(2 * A)
            penalties = torch.trace(B @ B_)

        return {
            "loss": -rewards + penalties,
            "rewards": rewards,
            "penalties": penalties,
        }
