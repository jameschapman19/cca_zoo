from abc import abstractmethod
from typing import Any, Iterable, List, Optional, Union, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback, EarlyStopping
from torch.utils import data
from torch.utils.data import DataLoader

from cca_zoo._base import BaseModel
from cca_zoo.data.deep import NumpyDataset
from cca_zoo.linear._iterative._base import _default_initializer

# Default Trainer kwargs
DEFAULT_TRAINER_KWARGS = dict(
    enable_checkpointing=False,
    logger=True,
    enable_model_summary=False,
    enable_progress_bar=True,
)

DEFAULT_LOADER_KWARGS = dict(
    num_workers=0, pin_memory=False, drop_last=False, shuffle=False
)


class BaseLoop(pl.LightningModule):
    """A base class for CCA loops.

    Attributes:
        weights (torch.nn.ParameterList): The CCA weights as torch parameters.
        tracking (bool): Whether to track the objective value during training.
        convergence_checking (bool): Whether to check the convergence condition during training.
        optimizer_kwargs (dict): The keyword arguments for the optimizer constructor.
        learning_rate (float): The learning rate for the optimizer.
    """

    def __init__(
        self,
        weights: Optional[List[np.ndarray]] = None,
        k: Optional[int] = None,
        tracking: bool = False,
        convergence_checking: bool = False,
        optimizer_kwargs: Optional[dict] = None,
        learning_rate: float = 1e-3,
    ) -> None:
        """Initialize the BaseLoop object.

        Args:
            weights (Optional[List[np.ndarray]], optional): The initial weights for the CCA loop, by default None
            k (Optional[int], optional): The current component, by default None
            tracking (bool, optional): Whether to track the objective value during training, by default False
            convergence_checking (bool, optional): Whether to check the convergence condition during training, by default False
            optimizer_kwargs (Optional[dict], optional): The keyword arguments for the optimizer constructor, by default None
            learning_rate (float, optional): The learning rate for the optimizer, by default 1e-3
        """
        super().__init__()
        if k is not None:
            self.weights = [weight[:, k] for weight in weights]
        else:
            self.weights = weights
        self.tracking = tracking
        self.convergence_checking = convergence_checking
        # Set the weights attribute as torch parameters with gradients
        self.weights = [
            torch.nn.Parameter(torch.from_numpy(weight), requires_grad=True)
            for weight in self.weights
        ]
        self.weights = torch.nn.ParameterList(self.weights)
        # Set the optimizer keyword arguments attribute with default values if none provided
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.learning_rate = learning_rate

    def forward(self, views: List[torch.Tensor]) -> List[torch.Tensor]:
        """Perform a forward pass on the input views.

        Args:
            views (List[torch.Tensor]): The input views as torch tensors.

        Returns:
            List[torch.Tensor]: The output views as torch tensors.
        """
        return [view @ weight for view, weight in zip(views, self.weights)]

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
            self.weights, lr=self.learning_rate, **kwargs
        )
        return optimizer

    def on_fit_end(self) -> None:
        """Perform some actions after the fit ends.

        For example, convert the weights from torch parameters to numpy arrays.
        """
        # if self.weights are torch parameters, convert them to numpy arrays
        if isinstance(self.weights, torch.nn.ParameterList):
            # weights to numpy arrays from torch parameters
            weights = [weight.detach().cpu().numpy() for weight in self.weights]
            del self.weights
            self.weights = weights

    def objective(self, *args, **kwargs) -> float:
        """Compute the objective value for the loop.

        This method should be implemented by subclasses.

        Returns:
            float: The objective value.

        Raises:
            NotImplementedError: If the method is not implemented by subclasses.
        """
        raise NotImplementedError


class BaseGradientModel(BaseModel):
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
        val_split=None,
        learning_rate=1,
        initialization: Union[str, callable] = "random",
        callbacks: Optional[Union[List[Callback], Callback]] = None,
        trainer_kwargs=None,
        convergence_checking=None,
        patience=10,
        track=None,
        verbose=None,
    ):
        super().__init__(
            latent_dimensions=latent_dimensions,
            copy_data=copy_data,
            random_state=random_state,
            accept_sparse=accept_sparse,
        )
        self.tol = tol
        self.batch_size = batch_size
        self.epochs = epochs
        # validate the split
        if val_split is not None:
            if val_split <= 0 or val_split >= 1:
                raise ValueError("Validation split must be between 0 and 1")
        self.val_split = val_split
        self.learning_rate = learning_rate
        # validate the initialization method
        if initialization not in ["random", "uniform", "unregularized", "pls"]:
            raise ValueError(
                "Initialization method must be one of ['random', 'uniform', 'unregularized', 'pls']"
            )
        else:
            self.initialization = initialization
        # validate the callbacks
        self.verbose = verbose
        self.patience = patience
        self.callbacks = callbacks or []
        # validate the convergence checking
        self.convergence_checking = convergence_checking
        # if convergence checking is a string
        if isinstance(self.convergence_checking, str):
            self.callbacks.append(
                ConvergenceCallback(
                    monitor=self.convergence_checking, min_delta=tol, patience=patience
                )
            )
        self.track = track
        if isinstance(self.track, str):
            self.callbacks.append(
                TrackingCallback(monitor=self.track, verbose=self.verbose)
            )
        self.dataloader_kwargs = dataloader_kwargs or DEFAULT_LOADER_KWARGS
        self.trainer_kwargs = trainer_kwargs or DEFAULT_TRAINER_KWARGS

    @abstractmethod
    def _get_pl_module(self, weights=None, k=None) -> BaseLoop:
        """Get model specific loop module for training.

        Parameters
        ----------
        weights : list of np.ndarray, optional
            The initial weights for the CCA module, by default None
        k : int, optional
            The current component, by default None

        Returns
        -------
        pl.LightningModule
            The CCA module object
        """
        raise NotImplementedError

    def fit(self, views: Iterable[np.ndarray], y=None, **kwargs):
        views = self._validate_data(views)
        self._check_params()
        self._initialize(views)
        self.weights = self._fit(views)
        return self

    def _fit(self, views: Iterable[np.ndarray]):
        train_dataloader, val_dataloader = self.get_dataloader(views)
        loop = self._get_pl_module(weights=self.weights)
        # make a trainer
        trainer = pl.Trainer(
            max_epochs=self.epochs,
            callbacks=self.callbacks,
            **self.trainer_kwargs,
        )
        trainer.fit(loop, train_dataloader, val_dataloader)
        # return the weights from the module. They will need to be changed from torch tensors to numpy arrays
        weights = loop.weights
        # if loop has tracked the objective, return the objective
        if hasattr(loop, "epoch_objective"):
            self.objective = loop.epoch_objective
        return weights

    def get_dataloader(
        self, views: Iterable[np.ndarray]
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        dataset = NumpyDataset(views) if self.batch_size else BatchNumpyDataset(views)

        collate_fn = None if self.batch_size else lambda x: x[0]

        if self.val_split:
            train_size = int((1 - self.val_split) * len(dataset))
            val_size = len(dataset) - train_size
            dataset, val_dataset = data.random_split(dataset, [train_size, val_size])
            val_loader = DataLoader(
                val_dataset,
                batch_size=val_size if not self.batch_size else self.batch_size,
                **self.dataloader_kwargs,
                collate_fn=collate_fn,
            )
        else:
            val_loader = None

        train_loader = DataLoader(
            dataset,
            batch_size=len(dataset) if not self.batch_size else self.batch_size,
            **self.dataloader_kwargs,
            collate_fn=collate_fn,
        )

        return train_loader, val_loader

    def _initialize(self, views: Iterable[np.ndarray]):
        """Initialize the CCA weights using the initialization method or function.

        Parameters
        ----------
        views : Iterable[np.ndarray]
            The input views to initialize the CCA weights from
        """
        pls = self._get_tags().get("pls", False)
        initializer = _default_initializer(
            self.initialization, self.random_state, self.latent_dimensions, pls
        )
        # Fit the initializer on the input views and get the weights as numpy arrays
        self.weights = initializer.fit(views).weights
        self.weights = [weights.astype(np.float32) for weights in self.weights]

    def _more_tags(self):
        # Indicate that this class is for multiview data
        return {"iterative": True}


class ConvergenceCallback(EarlyStopping):
    def __init__(
        self,
        monitor: str = "loss",
        min_delta: float = 0.0,
        patience: int = 0,
        verbose: bool = False,
        mode: str = "min",
        strict: bool = True,
        check_finite: bool = True,
    ):
        super().__init__(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            verbose=verbose,
            mode=mode,
            strict=strict,
            check_finite=check_finite,
        )
        if monitor == "weights_change":
            self.stopping_threshold = min_delta

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch: Any,
        batch_idx: int,
    ):
        pl_module.log(self.monitor, outputs[self.monitor].item())


class TrackingCallback(Callback):
    """
    Callback to track the objective function value during training
    """

    def __init__(self, monitor: str = "loss", verbose: bool = False):
        super().__init__()
        self.monitor = monitor
        self.verbose = verbose

    def on_train_start(self, trainer, pl_module):
        pl_module.batch_objective = []
        pl_module.epoch_objective = []

    def on_train_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        pl_module.batch_objective = []

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch: Any,
        batch_idx: int,
    ):
        pl_module.batch_objective.append(outputs[self.monitor].item())
        if self.verbose:
            # Print the objective function value and the current batch index
            print(
                f"Objective: {outputs[self.monitor].item():.3f} | Batch: {batch_idx}",
                end="\r",
            )

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        # epoch objective values are the mean of the batch objective values
        pl_module.epoch_objective.append(np.mean(pl_module.batch_objective))
        if self.verbose:
            # Print a new line after the last batch
            print()
            print(
                f"Objective: {pl_module.epoch_objective[-1]:.3f} | Epoch: {trainer.current_epoch}",
            )


class BatchNumpyDataset:
    def __init__(self, views, labels=None):
        self.views = [torch.from_numpy(view).float() for view in views]
        self.labels = torch.from_numpy(labels).float() if labels is not None else None

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if self.labels is not None:
            return {"views": self.views, "label": self.labels}
        else:
            return {"views": self.views}
