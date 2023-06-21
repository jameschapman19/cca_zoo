import numpy as np
import torch

from cca_zoo.classical._iterative._base import BaseLoop


class BaseGradientLoop(BaseLoop):
    def __init__(
        self,
        weights: list = None,
        k: int = None,
        learning_rate: float = 1e-3,
        optimizer_kwargs: dict = None,
        tracking: bool = False,
        convergence_checking: bool = False,
    ):
        """Initialize the gradient-based CCA loop.

        Parameters
        ----------
        weights : list, optional
            The initial weights for the CCA loop, by default None
        k : int, optional
            The index of the latent dimension to use for the CCA loop, by default None
        learning_rate : float, optional
            The learning rate for the optimizer, by default 1e-3
        optimizer_kwargs : dict, optional
            The keyword arguments for the optimizer creation, by default None
        """
        super().__init__(
            weights=weights,
            k=k,
            automatic_optimization=True,
            tracking=tracking,
            convergence_checking=convergence_checking,
        )
        # Set the weights attribute as torch parameters with gradients
        self.weights = [
            torch.nn.Parameter(torch.from_numpy(weight), requires_grad=True)
            for weight in self.weights
        ]
        self.weights = torch.nn.ParameterList(self.weights)

        # Set the optimizer keyword arguments attribute with default values if none provided
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.learning_rate = learning_rate

    def configure_optimizers(self):
        # construct optimizer using optimizer_kwargs
        optimizer_name = self.optimizer_kwargs.get("optimizer", "Adam")
        optimizer_kwargs = self.optimizer_kwargs.get("optimizer_kwargs", {})
        optimizer = getattr(torch.optim, optimizer_name)(
            self.weights, lr=self.learning_rate, **optimizer_kwargs
        )
        return optimizer

    def on_fit_end(self) -> None:
        # weights to numpy arrays from torch parameters
        weights = [weight.detach().cpu().numpy() for weight in self.weights]
        del self.weights
        self.weights = weights

    def forward(self, views):
        # if views are numpy arrays, convert to torch tensors
        if isinstance(views[0], np.ndarray):
            views = [torch.from_numpy(view) for view in views]
        return [view @ weight for view, weight in zip(views, self.weights)]
