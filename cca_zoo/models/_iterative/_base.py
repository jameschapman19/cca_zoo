from abc import ABC, abstractmethod
from typing import Iterable, List, Optional, Tuple, Union, Any

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback, EarlyStopping
from torch import Tensor
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm import tqdm

from cca_zoo.data.deep import NumpyDataset
from cca_zoo.models import MCCA, rCCA
from cca_zoo.models._base import BaseModel
from cca_zoo.models._dummy import DummyCCA

# Default Trainer kwargs
DEFAULT_TRAINER_KWARGS = dict(
    enable_checkpointing=False,
    logger=False,
    enable_model_summary=False,
    enable_progress_bar=False,
)

DEFAULT_LOADER_KWARGS = dict(
    num_workers=0, pin_memory=True, drop_last=False, shuffle=False
)


class BaseIterative(BaseModel):
    def __init__(
        self,
        latent_dims: int = 1,
        copy_data=True,
        random_state=None,
        tol=1e-3,
        deflation="cca",
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
        verbose=False,
    ):
        super().__init__(
            latent_dims=latent_dims,
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
        # validate the deflation method
        if deflation not in ["cca", "pls"]:
            raise ValueError("Deflation method must be one of ['cca','pls']")
        else:
            self.deflation = deflation
        # validate the initialization method
        if initialization not in ["random", "uniform", "pls", "cca"]:
            raise ValueError(
                "Initialization method must be one of ['random', 'uniform', 'pls', 'cca']"
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
    def _get_module(self, weights=None, k=None):
        """Get the CCA module for training.

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
        pass

    def fit(self, views: Iterable[np.ndarray], y=None, **kwargs):
        self._validate_data(views)
        self._check_params()
        self._initialize(views)
        self.weights = self._fit(views)
        return self

    def _fit(self, views: Iterable[np.ndarray]):
        train_dataloader, val_dataloader = self.get_dataloader(views)
        loop = self._get_module(weights=self.weights)
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

    def get_dataloader(self, views: Iterable[np.ndarray]):
        if self.batch_size is None:
            dataset = BatchNumpyDataset(views)
            collate_fn = lambda x: x[0]
        else:
            dataset = NumpyDataset(views)
            collate_fn = None
        if self.val_split is not None:
            train_size = int((1 - self.val_split) * len(dataset))
            val_size = len(dataset) - train_size
            dataset, val_dataset = data.random_split(dataset, [train_size, val_size])
            if self.batch_size is None:
                batch_size = val_size
            else:
                batch_size = self.batch_size
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                **self.dataloader_kwargs,
                collate_fn=collate_fn,
            )
        else:
            val_loader = None
        if self.batch_size is None:
            batch_size = len(dataset)
        else:
            batch_size = self.batch_size
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
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
        initializer = _default_initializer(
            self.initialization, self.random_state, self.latent_dims
        )
        # Fit the initializer on the input views and get the weights as numpy arrays
        self.weights = initializer.fit(views).weights
        self.weights = [weights.astype(np.float32) for weights in self.weights]


class BaseDeflation(BaseIterative, ABC):
    def _fit(self, views: Iterable[np.ndarray]):
        # tqdm for each latent dimension
        for k in tqdm(range(self.latent_dims), desc="Latent Dimension", leave=False):
            train_dataloader, val_dataloader = self.get_dataloader(views)
            loop = self._get_module(weights=self.weights, k=k)
            # make a trainer
            trainer = pl.Trainer(
                max_epochs=self.epochs, callbacks=self.callbacks, **self.trainer_kwargs
            )
            trainer.fit(loop, train_dataloader, val_dataloader)
            # return the weights from the module. They will need to be changed from torch tensors to numpy arrays
            weights = loop.weights
            for i, (view, weight) in enumerate(zip(views, weights)):
                self.weights[i][:, k] = weight
                views[i] = self._deflate(view, weight)
            # if loop has tracked the objective, return the objective
            if hasattr(loop, "epoch_objective"):
                self.objective = loop.epoch_objective
        return self.weights

    def _deflate(self, residual: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Deflate view residual by CCA deflation.

        Parameters
        ----------
        residual : np.ndarray
            The current residual data matrix for a view
        weights : np.ndarray
            The current CCA weights for a view

        Returns
        -------
        np.ndarray
            The deflated residual data matrix for a view

        Raises
        ------
        ValueError
            If deflation method is not one of ["cca", "pls"]
        """
        # Compute the score vector for a view
        score = residual @ weights

        # Deflate the residual by different methods based on the deflation attribute
        if self.deflation == "cca":
            return residual - np.outer(score, score) @ residual / np.dot(score, score)
        elif self.deflation == "pls":
            return residual - np.outer(score, weights)
        else:
            raise ValueError(
                f"Invalid deflation method: {self.deflation}. "
                f"Must be one of ['cca', 'pls']."
            )


class BaseLoop(pl.LightningModule):
    def __init__(
        self,
        weights=None,
        k=None,
        automatic_optimization=False,
        tracking=False,
        convergence_checking=False,
    ):
        super().__init__()
        if k is not None:
            self.weights = [weight[:, k] for weight in weights]
        else:
            self.weights = weights
        self.automatic_optimization = automatic_optimization
        self.tracking = tracking
        self.convergence_checking = convergence_checking

    def objective(self, *args, **kwargs) -> float:
        raise NotImplementedError

    def forward(self, views: list) -> list:
        """Compute the score vectors for each view.

        Parameters
        ----------
        views : list
            The input views to compute the score vectors from

        Returns
        -------
        list
            The score vectors for each view
        """
        return [view @ weight for view, weight in zip(views, self.weights)]

    def configure_optimizers(self):
        """
        We don't need an optimizer for manual optimization.
        """
        pass


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


def _default_initializer(initialization, random_state, latent_dims):
    if initialization == "random":
        initializer = DummyCCA(latent_dims, random_state=random_state, uniform=False)
    elif initialization == "uniform":
        initializer = DummyCCA(latent_dims, random_state=random_state, uniform=True)
    elif initialization == "pls":
        initializer = rCCA(latent_dims, random_state=random_state, c=1)
    elif initialization == "cca":
        initializer = MCCA(latent_dims)
    else:
        raise ValueError(
            "Initialization {type} not supported. Pass a generator implementing this method"
        )
    return initializer


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
        self.views = [view.astype(np.float32) for view in views]
        self.labels = labels if labels is not None else None

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if self.labels is not None:
            return {"views": self.views, "label": self.labels}
        else:
            return {"views": self.views}
