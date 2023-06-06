from typing import Union

import torch

from cca_zoo.models._base import PLSMixin
from cca_zoo.models._iterative._base import BaseIterative, BaseGradientLoop


class PLSStochasticPower(BaseIterative, PLSMixin):
    r"""
    A class used to fit Stochastic PLS

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
    Arora, Raman, et al. "Stochastic optimization for PCA and PLS." 2012 50th Annual Allerton Conference on Communication, Control, and Computing (Allerton). IEEE, 2012.

    """

    def __init__(
        self,
        latent_dims: int = 1,
        copy_data=True,
        random_state=None,
        tol=1e-4,
        accept_sparse=None,
        batch_size=20,
        epochs=1,
        learning_rate=1e-2,
        initialization: Union[str, callable] = "random",
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
        self.orth_required = False
        self.optimizer_kwargs = optimizer_kwargs
        if self.optimizer_kwargs is None:
            self.optimizer_kwargs = {}

    def _get_module(self, weights=None, k=None):
        return StochasticPowerLoopBase(
            weights=weights,
            k=k,
            learning_rate=self.learning_rate,
            optimizer_kwargs=self.optimizer_kwargs,
        )

    def _more_tags(self):
        return {"multiview": True, "stochastic": True}


class StochasticPowerLoopBase(BaseGradientLoop):
    def __init__(self, weights=None, k=None, learning_rate=1e-3, optimizer_kwargs=None):
        super().__init__(
            weights=weights,
            k=k,
            learning_rate=learning_rate,
            optimizer_kwargs=optimizer_kwargs,
        )

    def training_step(self, batch, batch_idx):
        for weight in self.weights:
            weight.data = self._orth(weight)
        scores = self(batch["views"])
        # find the pairwise covariance between the scores
        cov = torch.cov(torch.hstack(scores).T)
        loss = torch.trace(cov[: scores[0].shape[1], scores[0].shape[1] :])
        self.log("train_loss", loss)
        return loss

    def on_train_end(self) -> None:
        for weight in self.weights:
            weight.data = self._orth(weight)

    def configure_optimizers(self):
        # construct optimizer using optimizer_kwargs
        optimizer_name = self.optimizer_kwargs.get("optimizer", "Adam")
        optimizer_kwargs = self.optimizer_kwargs.get("optimizer_kwargs", {})
        optimizer = getattr(torch.optim, optimizer_name)(
            self.weights, lr=self.learning_rate, **optimizer_kwargs
        )
        return optimizer

    @staticmethod
    def _orth(U):
        Qu, Ru = torch.linalg.qr(U)
        Su = torch.sign(torch.sign(torch.diag(Ru)) + 0.5)
        return Qu @ torch.diag(Su)
