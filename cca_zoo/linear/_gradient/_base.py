from typing import Iterable, List, Union, Optional

import numpy as np
from cca_zoo._base import _BaseModel
from cca_zoo.linear._iterative._base import _default_initializer
from cca_zoo.linear._mcca import MCCA

DEFAULT_OPTIMIZER_KWARGS = dict(optimizer="SGD", nesterov=True, momentum=0.9)


class BaseGradientModel(_BaseModel):
    def __init__(
        self,
        latent_dimensions: int = 1,
        copy_data=True,
        random_state=None,
        tol=1e-3,
        accept_sparse=None,
        batch_size=None,
        epochs=1,
        learning_rate=5e-3,
        initialization: Union[str, callable] = "random",
        early_stopping=False,
        patience=5,
        nesterov=True,
        momentum=0.9,
        dampening=0.0,
    ):
        super().__init__(latent_dimensions, copy_data, accept_sparse, random_state)
        self.tol = tol
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.early_stopping = early_stopping
        self.patience = patience
        self.nesterov = nesterov
        self.momentum = momentum
        self.dampening = dampening
        # validate the initialization method
        if initialization not in ["random", "uniform", "unregularized", "pls"]:
            raise ValueError(
                "Initialization method must be one of ['random', 'uniform', 'unregularized', 'pls']"
            )
        else:
            self.initialization = initialization

    def __call__(self, views: Iterable[np.ndarray]):
        """Transform the input views using the learned weights_.

        Parameters
        ----------
        views : Iterable[np.ndarray]
            The input views to transform

        Returns
        -------
        List[np.ndarray]
            The transformed views
        """
        return self.transform(views)

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
        self._initialize(views)
        self.velocity = [np.zeros_like(weight) for weight in self.weights_]
        train_dataset, val_dataset = self.get_dataset(
            views, validation_views=validation_views
        )
        best_val_loss = np.inf
        for i in range(self.epochs):
            for batch, independent_batch in train_dataset:
                self.training_step(batch, independent_batch)
            if val_dataset is not None:
                val_loss = 0
                for batch, independent_batch in val_dataset:
                    val_loss += self.validation_step(batch, independent_batch)
                val_loss /= len(val_dataset)

                if self.early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_weights = self.weights_
                    else:
                        if i > self.patience:
                            self.weights_ = best_weights
                            break
        return self

    def get_dataset(self, views: Iterable[np.ndarray], validation_views=None):
        dataset = NumpyDataset(views, batch_size=self.batch_size)
        if validation_views is not None:
            val_dataset = NumpyDataset(validation_views, batch_size=self.batch_size)
        else:
            val_dataset = None
        return dataset, val_dataset

    def _initialize(self, views: Iterable[np.ndarray]):
        """Initialize the CCA weights_ using the initialization method or function.

        Parameters
        ----------
        views : Iterable[np.ndarray]
            The input representations to initialize the CCA weights_ from
        """
        pls = self._get_tags().get("pls", False)
        initializer = _default_initializer(
            self.initialization, self.random_state, self.latent_dimensions, pls
        )
        # Fit the initializer on the input representations and get the weights_ as numpy arrays
        self.weights_ = initializer.fit(views).weights_
        self.weights_ = [weights.astype(np.float32) for weights in self.weights_]

    def _more_tags(self):
        return {"multiview": True, "stochastic": True}

    def loss(
        self,
        representations: List[np.ndarray],
        independent_representations: Optional[List[np.ndarray]] = None,
    ):
        raise NotImplementedError

    def derivative(
        self,
        views: List[np.ndarray],
        representations: List[np.ndarray],
        independent_views: Optional[List[np.ndarray]] = None,
        independent_representations: Optional[List[np.ndarray]] = None,
    ):
        raise NotImplementedError

    def on_training_step_start(self):
        pass

    def training_step(self, batch, independent_batch=None):
        self.on_training_step_start()
        representations = self(batch)
        independent_representations = (
            self(independent_batch) if independent_batch is not None else None
        )
        loss = self.loss(representations, independent_representations)  # noqa
        manual_grads = self.derivative(
            batch,
            representations,
            independent_batch,
            independent_representations,
        )
        for i in range(len(self.weights_)):
            if self.nesterov:
                self.velocity[i] = (
                    -self.momentum * self.velocity[i]
                    + (1 - self.dampening) * manual_grads[i]
                )
                self.weights_[i] -= self.learning_rate * (
                    manual_grads[i] - self.velocity[i]
                )
            else:
                self.weights_[i] -= self.learning_rate * manual_grads[i]

    def validation_step(self, batch, independent_batch=None):
        representations = self(batch)
        independent_representations = (
            self(independent_batch) if independent_batch is not None else None
        )
        loss = self.loss(representations, independent_representations)
        return loss["objective"]

    def correlation_captured(self, z):
        # Remove mean from each view
        z = [zi - zi.mean(0) for zi in z]
        return MCCA(latent_dimensions=self.latent_dimensions).fit(z).score(z).sum()


class NumpyDataset:
    def __init__(self, views, batch_size=None, shuffle=True, random_state=None):
        self.views = [np.array(view, dtype=np.float32) for view in views]
        self.batch_size = batch_size if batch_size is not None else len(self.views[0])
        self.shuffle = shuffle
        self.random_state = np.random.RandomState(random_state)
        self.indices = np.arange(len(self.views[0]))

    def __iter__(self):
        if self.shuffle:
            self.random_state.shuffle(self.indices)
        self.n_batches = len(self) // self.batch_size
        self.i = 0
        return self

    def __next__(self):
        if self.i < self.n_batches:
            indices = self.indices[
                self.i * self.batch_size : (self.i + 1) * self.batch_size
            ]
            batch = [view[indices] for view in self.views]

            # Generate random indices for the independent batch
            independent_indices = self.random_state.randint(
                0, len(self.views[0]), self.batch_size
            )
            independent_batch = [view[independent_indices] for view in self.views]

            self.i += 1
            return batch, independent_batch
        else:
            raise StopIteration

    def __len__(self):
        return len(self.views[0])
