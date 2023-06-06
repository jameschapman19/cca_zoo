from typing import Union

import torch
import numpy as np
from cca_zoo.models._base import PLSMixin
from cca_zoo.models._iterative._base import BaseIterative, BaseGradientLoop


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
        epochs=1,
        learning_rate=1e-1,
        initialization: Union[str, callable] = "random",
        dataloader_kwargs=None,
        optimizer_kwargs=None,
        convergence_checking=False,
        patience=10,
    ):
        self.previous_views = None
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
        )

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
                "objective": torch.tensor(0, requires_grad=True, dtype=torch.float32),
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
        return loss["objective"]

    def get_AB(self, z):
        # Getting the dimensions of the encoded views
        N, D = z[0].size()
        # Computing the covariance matrix of the concatenated views
        C = torch.cov(torch.hstack(z).T)
        # Extracting the cross-covariance and auto-covariance matrices
        A = C[:D, D:] + C[D:, :D]
        if self.objective == "cca":
            B = C[:D, :D] + C[D:, D:]
        elif self.objective == "pls":
            B = (
                self.weights[0].T @ self.weights[0]
                + self.weights[1].T @ self.weights[1]
            )
        return A, B

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
            "objective": -rewards + penalties,
            "rewards": rewards,
            "penalties": penalties,
        }
