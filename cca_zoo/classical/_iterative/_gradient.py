import torch

from cca_zoo.classical._iterative._base import BaseLoop


class GradientLoop(BaseLoop):
    def __init__(
        self,
        weights=None,
        k=None,
        tracking=False,
        convergence_checking=False,
        optimizer_kwargs=None,
        learning_rate=1e-3,
    ):
        super().__init__(
            weights=weights,
            k=k,
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
        self.automatic_optimization = True

    def configure_optimizers(self):
        # construct optimizer using optimizer_kwargs
        optimizer_name = self.optimizer_kwargs.get("optimizer", "Adam")
        optimizer_kwargs = self.optimizer_kwargs.get("optimizer_kwargs", {})
        optimizer = getattr(torch.optim, optimizer_name)(
            self.weights, lr=self.learning_rate, **optimizer_kwargs
        )
        return optimizer

    def on_fit_end(self) -> None:
        # if self.weights are torch parameters, convert them to numpy arrays
        if isinstance(self.weights, torch.nn.ParameterList):
            # weights to numpy arrays from torch parameters
            weights = [weight.detach().cpu().numpy() for weight in self.weights]
            del self.weights
            self.weights = weights
