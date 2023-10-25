from typing import Iterable, List, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from cca_zoo._base import BaseModel
from cca_zoo.deep.utils import NumpyDataset
from cca_zoo.linear._iterative._base import _default_initializer

# Default Trainer kwargs
DEFAULT_TRAINER_KWARGS = dict(
    enable_checkpointing=False,
    logger=False,
    enable_model_summary=False,
    enable_progress_bar=True,
)

DEFAULT_LOADER_KWARGS = dict(pin_memory=False, drop_last=True, shuffle=True)

DEFAULT_OPTIMIZER_KWARGS = dict(optimizer="Adam")


class BaseGradientModel(BaseModel, pl.LightningModule):
    def __init__(
        self,
        latent_dimensions: int = 1,
        copy_data=True,
        random_state=None,
        tol=1e-3,
        accept_sparse=None,
        batch_size=None,
        dataloader_kwargs=None,
        epochs=1,
        learning_rate=1e-1,
        initialization: Union[str, callable] = "random",
        trainer_kwargs=None,
        optimizer_kwargs=None,
        early_stopping=False,
    ):
        BaseModel.__init__(
            self,
            latent_dimensions=latent_dimensions,
            copy_data=copy_data,
            random_state=random_state,
            accept_sparse=accept_sparse,
        )
        pl.LightningModule.__init__(self)
        self.tol = tol
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        # validate the initialization method
        if initialization not in ["random", "uniform", "unregularized", "pls"]:
            raise ValueError(
                "Initialization method must be one of ['random', 'uniform', 'unregularized', 'pls']"
            )
        else:
            self.initialization = initialization
        self.dataloader_kwargs = dataloader_kwargs or DEFAULT_LOADER_KWARGS
        self.trainer_kwargs = trainer_kwargs or DEFAULT_TRAINER_KWARGS
        self.optimizer_kwargs = optimizer_kwargs or DEFAULT_OPTIMIZER_KWARGS
        self.early_stopping = early_stopping

    def fit(self, views: Iterable[np.ndarray], y=None, validation_views=None, **kwargs):
        views = self._validate_data(views)
        if validation_views is not None:
            validation_views = self._validate_data(validation_views)
        self._check_params()
        self.weights_ = self._fit(views, validation_views=validation_views)
        return self

    def _fit(self, views: Iterable[np.ndarray], validation_views=None):
        self._initialize(views)
        # Set the weights_ attribute as torch parameters with gradients
        self.torch_weights = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.from_numpy(weight), requires_grad=True)
                for weight in self.weights_
            ]
        )
        trainer = pl.Trainer(
            max_epochs=self.epochs,
            **self.trainer_kwargs,
        )
        train_dataset, val_dataset = self.get_dataset(
            views, validation_views=validation_views
        )
        train_dataloader, val_dataloader = self.get_dataloader(
            train_dataset, val_dataset
        )
        trainer.fit(self, train_dataloader, val_dataloader)
        # return the weights_ from the module. They will need to be changed from torch tensors to numpy arrays
        weights = [weight.detach().cpu().numpy() for weight in self.torch_weights]
        return weights

    def get_dataset(self, views: Iterable[np.ndarray], validation_views=None):
        dataset = NumpyDataset(views)
        if validation_views is not None:
            val_dataset = NumpyDataset(validation_views)
        else:
            val_dataset = None
        return dataset, val_dataset

    def get_dataloader(self, train_dataset, val_dataset):
        train_loader = DataLoader(
            train_dataset,
            batch_size=len(train_dataset)
            if self.batch_size is None
            else self.batch_size,
            **self.dataloader_kwargs,
        )
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=len(val_dataset)
                if self.batch_size is None
                else self.batch_size,
                **self.dataloader_kwargs,
            )
        else:
            val_loader = None
        return train_loader, val_loader

    def _initialize(self, views: Iterable[np.ndarray]):
        """Initialize the CCALoss weights_ using the initialization method or function.

        Parameters
        ----------
        views : Iterable[np.ndarray]
            The input representations to initialize the CCALoss weights_ from
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

    def forward(self, views: List[torch.Tensor]) -> List[torch.Tensor]:
        """Perform a forward pass on the input representations.

        Args:
            views (List[torch.Tensor]): The input representations as torch tensors.

        Returns:
            List[torch.Tensor]: The output representations as torch tensors.
        """
        return [view @ weight for view, weight in zip(views, self.torch_weights)]

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer for the loop.

        Returns:
            torch.optim.Optimizer: The optimizer object.
        """
        # construct optimizer using optimizer_kwargs
        optimizer_name = self.optimizer_kwargs.get("optimizer", "Adam")
        kwargs = self.optimizer_kwargs.copy()
        kwargs.pop("optimizer", None)
        optimizer = getattr(torch.optim, optimizer_name)(
            self.torch_weights, lr=self.learning_rate, **kwargs
        )
        return optimizer

    def objective(self, *args, **kwargs) -> float:
        """Compute the objective value for the loop.

        This method should be implemented by subclasses.

        Returns:
            float: The objective value.

        Raises:
            NotImplementedError: If the method is not implemented by subclasses.
        """
        raise NotImplementedError
