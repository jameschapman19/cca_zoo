"""
This is a wrapper class for DCCA_base models.

It inherits from the CCA_Base class which allows us to borrow some of the functionality. In particular,
we could borrow the gridsearch_fit method to search the hyperparameter space.
"""

import copy
import itertools

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import cca_zoo.plot_utils
from cca_zoo.dcca import DCCA_base
from cca_zoo.wrappers import CCA_Base

class DeepWrapper(CCA_Base):

    def __init__(self, model: DCCA_base, device: str = 'cuda'):
        super().__init__(latent_dims=model.latent_dims)
        self.model = model
        self.device = device
        if not torch.cuda.is_available() and self.device == 'cuda':
            self.device = 'cpu'
        self.latent_dims = model.latent_dims

    def fit(self, *views, labels=None, val_split=0.2, batch_size=0, patience=0, epochs=1, train_correlations=True):
        """

        :param views: EITHER 2D numpy arrays for each view separated by comma with the same number of rows (nxp)
                        OR torch.torch.utils.data.Dataset
                        OR 2 or more torch.utils.data.Subset separated by commas
        :param labels:
        :param val_split: the ammount of data used for validation
        :param batch_size: the minibatch size
        :param patience: if 0 train to num_epochs, else if validation score doesn't improve after patience epochs stop training
        :param epochs: maximum number of epochs to train
        :param train_correlations: if True generate training correlations
        :return:
        """
        self.batch_size = batch_size
        if type(views[0]) is np.ndarray:
            dataset = cca_zoo.data.CCA_Dataset(*views, labels=labels)
            ids = np.arange(len(dataset))
            np.random.shuffle(ids)
            train_ids, val_ids = np.array_split(ids, 2)
            val_dataset = Subset(dataset, val_ids)
            train_dataset = Subset(dataset, train_ids)
        elif isinstance(views[0], torch.utils.data.Dataset):
            if len(views) == 2:
                assert (isinstance(views[0], torch.utils.data.Subset))
                assert (isinstance(views[1], torch.utils.data.Subset))
                train_dataset, val_dataset = views[0], views[1]
            else:
                dataset = views[0]
                lengths = [len(dataset) - int(len(dataset) * val_split), int(len(dataset) * val_split)]
                train_dataset, val_dataset = torch.utils.data.random_split(dataset, lengths)

        if batch_size == 0:
            train_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset), drop_last=True)
            val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset))
        else:
            train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, drop_last=True)
            val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size)

        # First we get the model class.
        # These have a forward method which takes data inputs and outputs the variables needed to calculate their
        # respective loss. The models also have loss functions as methods but we can also customise the loss by calling
        # a_loss_function(model(data))
        num_params = sum(p.numel() for p in self.model.parameters())
        print('total parameters: ', num_params)
        best_model = copy.deepcopy(self.model.state_dict())
        self.model.float().to(self.device)
        min_val_loss = torch.tensor(np.inf)
        epochs_no_improve = 0
        early_stop = False
        all_train_loss = []
        all_val_loss = []

        for epoch in range(1, epochs + 1):
            if not early_stop:
                epoch_train_loss = self.train_epoch(train_dataloader)
                print('====> Epoch: {} Average train loss: {:.4f}'.format(
                    epoch, epoch_train_loss))
                epoch_val_loss = self.val_epoch(val_dataloader)
                print('====> Epoch: {} Average val loss: {:.4f}'.format(
                    epoch, epoch_val_loss))

                if epoch_val_loss < min_val_loss or epoch == 1:
                    min_val_loss = epoch_val_loss
                    best_model = copy.deepcopy(self.model.state_dict())
                    print('Min loss %0.2f' % min_val_loss)
                    epochs_no_improve = 0

                else:
                    epochs_no_improve += 1
                    # Check early stopping condition
                    if epochs_no_improve == patience and patience > 0:
                        print('Early stopping!')
                        early_stop = True
                        self.model.load_state_dict(best_model)

                all_train_loss.append(epoch_train_loss)
                all_val_loss.append(epoch_val_loss)
        cca_zoo.plot_utils.plot_training_loss(all_train_loss, all_val_loss)
        if train_correlations:
            self.train_correlations = self.predict_corr(train_dataset, train=True)
        return self

    def train_epoch(self, train_dataloader: torch.utils.data.DataLoader):
        """
        Train a single epoch
        :param train_dataloader: a dataloader for training data
        :return: average loss over the epoch
        """
        self.model.train()
        train_loss = 0
        for batch_idx, (data, label) in enumerate(train_dataloader):
            data = [d.float().to(self.device) for d in list(data)]
            loss = self.model.update_weights(*data)
            train_loss += loss.item()
        return train_loss / len(train_dataloader)

    def val_epoch(self, val_dataloader: torch.utils.data.DataLoader):
        """
        Validate a single epoch
        :param val_dataloader: a dataloder for validation data
        :return: average validation loss over the epoch
        """
        self.model.eval()
        with torch.no_grad():
            total_val_loss = 0
            for batch_idx, (data, label) in enumerate(val_dataloader):
                data = [d.float().to(self.device) for d in list(data)]
                loss = self.model.loss(*data)
                total_val_loss += loss.item()
        return total_val_loss / len(val_dataloader)

    def predict_corr(self, *views, train=False):
        """
        :param views: EITHER numpy arrays separated by comma. Each view needs to have the same number of features as its
         corresponding view in the training data
                        OR torch.torch.utils.data.Dataset
                        OR 2 or more torch.utils.data.Subset separated by commas
        :return: numpy array containing correlations between each pair of views for each dimension (#views*#views*#latent_dimensions)
        """
        transformed_views = self.transform(*views, train=train)
        all_corrs = []
        for x, y in itertools.product(transformed_views, repeat=2):
            all_corrs.append(np.diag(np.corrcoef(x.T, y.T)[:self.latent_dims, self.latent_dims:]))
        all_corrs = np.array(all_corrs).reshape(
            (len(transformed_views), len(transformed_views), self.latent_dims))
        return all_corrs

    def transform(self, *views, labels=None, train=False):
        if type(views[0]) is np.ndarray:
            test_dataset = cca_zoo.data.CCA_Dataset(*views, labels=labels)
        elif isinstance(views[0], torch.utils.data.Dataset):
            test_dataset = views[0]
        if self.batch_size > 0:
            test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size)
        else:
            test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset))
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(test_dataloader):
                data = [d.float().to(self.device) for d in list(data)]
                z = self.model(*data)
                if batch_idx == 0:
                    z_list = [z_i.detach().cpu().numpy() for i, z_i in enumerate(z)]
                else:
                    z_list = [np.append(z_list[i], z_i.detach().cpu().numpy(), axis=0) for
                              i, z_i in enumerate(z)]
        # For trace-norm objective models we need to apply a linear CCA to outputs
        if self.model.post_transform:
            if train:
                self.cca = cca_zoo.wrappers.MCCA(latent_dims=self.latent_dims)
                self.cca.fit(*z_list)
                z_list = self.cca.transform(*z_list)
            else:
                z_list = self.cca.transform(*z_list)
        return z_list

    def predict_view(self, *views, labels=None):
        if type(views[0]) is np.ndarray:
            test_dataset = cca_zoo.data.CCA_Dataset(*views, labels=labels)
        elif isinstance(views[0], torch.utils.data.Dataset):
            test_dataset = views[0]

        test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset))
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(test_dataloader):
                data = [d.float().to(self.device) for d in list(data)]
                x = self.model.recon(*data)
                if batch_idx == 0:
                    x_list = [x_i.detach().cpu().numpy() for i, x_i in enumerate(x)]
                else:
                    x_list = [np.append(x_list[i], x_i.detach().cpu().numpy(), axis=0) for
                              i, x_i in enumerate(x)]
        return x_list
