from typing import Iterable, List, Union

import numpy as np
import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping
from torch.utils.data import DataLoader
from cca_zoo.linear._mcca import MCCA
from cca_zoo._base import _BaseModel
from cca_zoo.deep.data import NumpyDataset
from cca_zoo.linear._iterative._base import _default_initializer
import warnings
# Default Trainer kwargs
DEFAULT_TRAINER_KWARGS = dict(
    enable_checkpointing=False,
    logger=False,
    enable_model_summary=False,
    enable_progress_bar=False,
    accelerator="cpu",
)

DEFAULT_LOADER_KWARGS = dict(pin_memory=False, drop_last=True, shuffle=True)

DEFAULT_OPTIMIZER_KWARGS = dict(optimizer="SGD", nesterov=True, momentum=0.9)


class BaseGradientModel(_BaseModel, pl.LightningModule):
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
        learning_rate=5e-3,
        initialization: Union[str, callable] = "random",
        optimizer_kwargs=None,
        early_stopping=True,
        logging=False,
    ):
        _BaseModel.__init__(
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
        self.optimizer_kwargs = optimizer_kwargs or DEFAULT_OPTIMIZER_KWARGS
        self.early_stopping = early_stopping
        if early_stopping:
            if not logging:
                warnings.warn("Early stopping is enabled. Logging is automatically enabled.", RuntimeWarning)
            logging = True
        self.logging = logging


    def fit(
        self,
        views: Iterable[np.ndarray],
        y=None,
        validation_views=None,
        **trainer_kwargs
    ):
        views = self._validate_data(views)
        if validation_views is not None:
            validation_views = self._validate_data(validation_views)
        self._check_params()
        self.weights_ = self._fit(
            views, validation_views=validation_views, **trainer_kwargs
        )
        return self

    def _fit(
        self, views: Iterable[np.ndarray], validation_views=None, **trainer_kwargs
    ):
        self._initialize(views)
        # Set the weights_ attribute as torch parameters with gradients
        self.torch_weights = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.from_numpy(weight), requires_grad=True)
                for weight in self.weights_
            ]
        )
        # if self.early_stopping:
        #     # Define the EarlyStopping callback
        #     early_stop_callback = EarlyStopping(
        #         monitor='train/objective',  # Metric to monitor
        #         min_delta=0.05,  # Minimum change to qualify as an improvement
        #         patience=3,  # Number of epochs with no improvement after which training will be stopped
        #     )
        trainer = pl.Trainer(
            max_epochs=self.epochs,
            # callbacks=early_stop_callback if self.early_stopping else None,
            # if trainer_kwargs is not None trainer_kwargs will override the defaults
            **{**DEFAULT_TRAINER_KWARGS, **trainer_kwargs},
        )
        train_dataset, val_dataset = self.get_dataset(
            views, validation_views=validation_views
        )
        train_dataloader, val_dataloader = self.get_dataloader(
            train_dataset, val_dataset
        )
        trainer.fit(self, train_dataloader, val_dataloader)
        # return the weights_ from the module. They will need to be changed from torch tensors to numpy arrays
        weights = [weight.detach().numpy() for weight in self.torch_weights]
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
            self.dataloader_kwargs["shuffle"] = False
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

    def on_train_epoch_end(self) -> None:
        scheduler = self.lr_schedulers()
        scheduler.step()

    def _initialize(self, views: Iterable[np.ndarray]):
        """Initialize the _CCALoss weights_ using the initialization method or function.

        Parameters
        ----------
        views : Iterable[np.ndarray]
            The input representations to initialize the _CCALoss weights_ from
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
        optimizer_name = self.optimizer_kwargs.get("optimizer")
        kwargs = self.optimizer_kwargs.copy()
        kwargs.pop("optimizer", None)
        optimizer = getattr(torch.optim, optimizer_name)(
            self.torch_weights, lr=self.learning_rate, **kwargs
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0 if self.batch_size is None else 0.9)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def objective(self, *args, **kwargs) -> float:
        """Compute the objective value for the loop.

        This method should be implemented by subclasses.

        Returns:
            float: The objective value.

        Raises:
            NotImplementedError: If the method is not implemented by subclasses.
        """
        raise NotImplementedError

    def correlation_captured(self, z):
        # Remove mean from each view
        z = [zi - zi.mean(0) for zi in z]
        return MCCA(latent_dimensions=self.latent_dimensions).fit(z).score(z).sum()

    def score(self, loader: torch.utils.data.DataLoader, **kwargs):
        z = self.transform(loader)
        corr = self.correlation_captured(z)
        return corr
